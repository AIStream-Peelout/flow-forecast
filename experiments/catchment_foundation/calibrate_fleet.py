"""
Fleet-wide classical GR4-snow calibration over a basin manifest (warm-start stage one).

For every non-holdout basin this script assembles the same physical forcing the hybrid model
sees ([P_grid mm/hr, PET mm/hr, lapse-corrected T degC, SW]) from the basin's hourly CSV,
converts observed cfs to mm/hr, and runs
:func:`~flood_forecast.ode.physics.gr4_calibration.calibrate_gr4_snow` on the most recent
training-period years. Results append to a JSON that stage two
(``pretrain_parameter_head.py``) uses as regression targets, and which is independently
interesting as a per-basin "is rigid GR4 fittable here at all" census (regulated basins
should stand out with low calibrated NSE).

The fixed-step Euler simulator differs slightly from the ODE solver used in end-to-end
training; calibrated parameters are warm-start targets, not final estimates, so this
approximation is acceptable.

Resumable: basins already present in the output JSON are skipped.
"""
import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

TRAIN_END = "2022-01-01"


def build_basin_series(basin: Dict, preprocessing: Dict, train_end: str,
                       n_years: float) -> Optional[Dict]:
    """
    Assembles calibration forcing and observed flow for one basin.

    :param basin: The basin's manifest entry (csv_path, area_sq_km, temp_offset_c).
    :type basin: Dict
    :param preprocessing: The manifest preprocessing section (used for column names only).
    :type preprocessing: Dict
    :param train_end: Exclusive upper date bound (the training split end).
    :type train_end: str
    :param n_years: Length of the calibration window ending at ``train_end``.
    :type n_years: float
    :return: Dict with ``forcing`` (n,4 tensor), ``observed`` (n, mm/hr) and ``coverage``,
        or None if the record ends before the calibration window.
    :rtype: Dict, optional
    """
    frame = pd.read_csv(basin["csv_path"], parse_dates=["datetime"])
    frame = frame.set_index("datetime").sort_index()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    end = pd.Timestamp(train_end)
    start = end - pd.Timedelta(days=int(365.25 * n_years))
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    if len(frame) < 24 * 365:
        return None
    needed = ["precipitation", "pet_mm_hr", "temperature", "shortwave_radiation", "cfs"]
    frame = frame[needed].apply(pd.to_numeric, errors="coerce")
    frame = frame.resample("1h").mean()
    precip = frame["precipitation"].to_numpy(dtype=np.float32)
    pet = frame["pet_mm_hr"].to_numpy(dtype=np.float32)
    temp_c = frame["temperature"].to_numpy(dtype=np.float32) - 273.15 \
        + float(basin.get("temp_offset_c", 0.0))
    shortwave = frame["shortwave_radiation"].to_numpy(dtype=np.float32)
    observed = frame["cfs"].to_numpy(dtype=np.float32) * 0.0283168 * 3.6 \
        / float(basin["area_sq_km"])
    # Forcing gaps are interpolated conservatively (precipitation missing -> zero, state
    # channels ffilled); observed-flow gaps stay NaN and are masked out of the objective.
    precip = np.nan_to_num(precip, nan=0.0)
    pet = pd.Series(pet).ffill().bfill().fillna(0.0).to_numpy()
    temp_c = pd.Series(temp_c).interpolate(limit=24).ffill().bfill().to_numpy()
    shortwave = pd.Series(shortwave).ffill().bfill().fillna(0.0).to_numpy()
    forcing = torch.from_numpy(np.stack([precip, pet, temp_c, shortwave], axis=-1))
    observed_tensor = torch.from_numpy(observed)
    coverage = float(np.isfinite(observed).mean())
    return {"forcing": forcing, "observed": observed_tensor, "coverage": coverage}


def main() -> None:
    """
    CLI entry point for fleet calibration.

    :return: None
    :rtype: None
    """
    from flood_forecast.ode.physics.gr4_calibration import calibrate_gr4_snow

    parser = argparse.ArgumentParser(description="Classical GR4-snow fleet calibration")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="Path of the results JSON")
    parser.add_argument("--n-years", type=float, default=3.0,
                        help="Calibration window length ending at --train-end")
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--n-random", type=int, default=512)
    parser.add_argument("--refine-rounds", type=int, default=3)
    parser.add_argument("--include-holdout", action="store_true",
                        help="Also calibrate final-holdout basins (diagnostics only; stage two "
                             "must never train on them)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    basins = [b for b in manifest["basins"]
              if args.include_holdout or b.get("split") != "holdout"]
    print("calibrating %d basins (%d already done)" % (len(basins), len(results)))
    for index, basin in enumerate(basins):
        site = basin["site_id"]
        if site in results:
            continue
        series = build_basin_series(basin, manifest.get("preprocessing", {}),
                                    args.train_end, args.n_years)
        if series is None:
            results[site] = {"status": "insufficient_record"}
        else:
            try:
                calibration = calibrate_gr4_snow(
                    series["forcing"], series["observed"], n_random=args.n_random,
                    refine_rounds=args.refine_rounds, seed=args.seed)
                calibration.update({"status": "ok", "split": basin.get("split"),
                                    "coverage": series["coverage"]})
                results[site] = calibration
            except ValueError as error:
                results[site] = {"status": "error", "error": str(error)}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)
        entry = results[site]
        print("[%d/%d] %s -> %s (NSE %.3f vs midpoint %.3f)"
              % (index + 1, len(basins), site, entry["status"],
                 entry.get("nse", float("nan")),
                 entry.get("midpoint_nse", float("nan"))), flush=True)
    ok = [r for r in results.values() if r.get("status") == "ok"]
    if ok:
        nses = sorted(r["nse"] for r in ok)
        print("done: %d ok; median calibrated NSE %.3f; %d basins beat midpoint"
              % (len(ok), nses[len(nses) // 2],
                 sum(1 for r in ok if r["nse"] > r["midpoint_nse"])))


if __name__ == "__main__":
    main()
