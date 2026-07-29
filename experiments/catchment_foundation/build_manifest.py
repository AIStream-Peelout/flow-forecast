"""
Builds the multi-basin manifest for catchment foundation-model training.

Reads the Water repo's CO scrape registry, keeps gauges marked "completed" with usable static
attributes, computes per-basin train-period normalization stats (met means/stds and the flow
standard deviation in mm/hr), assigns the ungauged-generalization holdout, and writes the manifest
JSON consumed by ``MultiBasinWindowLoader`` and ``HybridGR4MultiBasin``.

Usage::

    python experiments/catchment_foundation/build_manifest.py \
        --out experiments/catchment_foundation/manifests/co_manifest.json
"""
import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

WATER_CO = os.path.expanduser("~/Documents/GitHub/Water/pilot_data/scrapes/CO")
SNODAS_DIR = os.path.expanduser("~/Documents/GitHub/Water/pilot_data/snodas_series/CO")
EMBEDDING_PATH = os.path.expanduser(
    "~/Documents/GitHub/Water/pilot_data/embedding_dataset/CO/embeddings_concat.pt")
CFS_TO_MM_HR = 0.0283168 * 3.6  # multiply by 1/area_km2 for mm/hr
TRAIN_END = "2022-01-01"  # scale stats and training data end here (valid=2022, test=2023+)
MET_COLS = ["precipitation", "temperature", "shortwave_radiation", "longwave_radiation",
            "specific_humidity", "wind_east", "wind_north", "p01m", "pet_mm_hr"]


def build_basin_entry(site_id: str, embedding_sites: set, min_train_hours: int) -> Optional[Dict]:
    """
    Builds one basin's manifest entry, or None when the gauge is unusable.

    :param site_id: The USGS site id.
    :type site_id: str
    :param embedding_sites: Site ids present in the pretrained embedding bank.
    :type embedding_sites: set
    :param min_train_hours: Minimum valid pre-cutoff flow hours required to keep the gauge.
    :type min_train_hours: int
    :return: The manifest entry dict, or None (with a printed reason).
    :rtype: Optional[Dict]
    """
    static_path = os.path.join(WATER_CO, site_id, site_id + "_static.json")
    csv_path = os.path.join(WATER_CO, site_id, site_id + "_hourly_full.csv")
    if not (os.path.exists(static_path) and os.path.exists(csv_path)):
        print("skip %s: missing files" % site_id)
        return None
    with open(static_path) as f:
        static = json.load(f)
    area = static.get("gages2_DRAIN_SQKM")
    if not area or area <= 0:
        drain_sq_mi = static.get("drain_area_va")
        area = float(drain_sq_mi) * 2.58999 if drain_sq_mi else None
    if not area or area <= 0:
        print("skip %s: no drainage area" % site_id)
        return None
    elev_basin = static.get("gages2_ELEV_MEAN_M_BASIN")
    alt_ft = static.get("alt_va")
    temp_offset_c = 0.0
    if elev_basin is not None and alt_ft is not None:
        gauge_m = float(alt_ft) * 0.3048
        temp_offset_c = -6.5 * (float(elev_basin) - gauge_m) / 1000.0
    header = pd.read_csv(csv_path, nrows=0).columns
    present = [col for col in MET_COLS if col in header]
    required = [col for col in MET_COLS if col != "p01m"]  # p01m is fillable from NLDAS precip
    missing = [col for col in required if col not in header]
    if missing:
        print("skip %s: missing met columns %s" % (site_id, missing))
        return None
    frame = pd.read_csv(csv_path, usecols=["datetime", "cfs"] + present)
    if "p01m" not in frame.columns:
        frame["p01m"] = np.nan
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True).dt.tz_localize(None)
    train = frame[frame["datetime"] < pd.Timestamp(TRAIN_END)]
    flow_mm = train["cfs"].to_numpy() * CFS_TO_MM_HR / area
    valid_hours = int(np.isfinite(flow_mm).sum())
    if valid_hours < min_train_hours:
        print("skip %s: only %d valid pre-%s flow hours" % (site_id, valid_hours, TRAIN_END))
        return None
    flow_scale = float(np.nanstd(flow_mm))
    if not np.isfinite(flow_scale) or flow_scale <= 0:
        print("skip %s: degenerate flow variance" % site_id)
        return None
    filled_p01m = train["p01m"].fillna(train["precipitation"])
    met_stats = {}
    for col in MET_COLS:
        series = filled_p01m if col == "p01m" else train[col]
        mean, std = float(series.mean()), float(series.std())
        if not (np.isfinite(mean) and np.isfinite(std)) or std <= 0:
            print("skip %s: degenerate met column %s" % (site_id, col))
            return None
        met_stats[col] = [mean, std]
    rows_2023 = int((frame["datetime"] >= pd.Timestamp("2023-01-01")).sum())
    entry = {"site_id": site_id, "csv_path": csv_path, "area_sq_km": float(area),
             "temp_offset_c": round(temp_offset_c, 4), "flow_scale_mm_hr": flow_scale,
             "met_stats": met_stats, "has_embedding": site_id in embedding_sites,
             "train_valid_hours": valid_hours, "rows_2023_plus": rows_2023, "split": "train"}
    swe_csv = os.path.join(SNODAS_DIR, site_id + "_snodas_swe.csv")
    if os.path.exists(swe_csv):
        entry["swe_csv_path"] = swe_csv
    return entry


def main() -> None:
    """
    Builds and writes the manifest.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output manifest JSON path")
    parser.add_argument("--n-holdout", type=int, default=15,
                        help="Basins held out entirely for the ungauged-generalization test")
    parser.add_argument("--min-train-years", type=float, default=2.0,
                        help="Minimum valid pre-cutoff years of flow to keep a gauge")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(os.path.join(WATER_CO, "registry.json")) as f:
        registry = json.load(f)
    completed = sorted(k for k, v in registry.items() if v.get("status") == "completed")
    bank = torch.load(EMBEDDING_PATH, weights_only=True)
    embedding_sites = set(bank["site_ids"])
    print("%d completed gauges, %d embedding sites" % (len(completed), len(embedding_sites)))

    basins = []
    for site_id in completed:
        entry = build_basin_entry(site_id, embedding_sites,
                                  int(args.min_train_years * 365 * 24))
        if entry is not None:
            basins.append(entry)

    # Ungauged holdout: embedded basins with 2023+ data so the split is actually evaluable.
    eligible = [b["site_id"] for b in basins
                if b["has_embedding"] and b["rows_2023_plus"] > 24 * 120]
    rng = np.random.default_rng(args.seed)
    holdout = set(rng.choice(eligible, size=min(args.n_holdout, len(eligible)),
                             replace=False).tolist())
    for basin in basins:
        if basin["site_id"] in holdout:
            basin["split"] = "holdout"

    manifest = {
        "embedding_path": EMBEDDING_PATH,
        "train_end": TRAIN_END,
        "preprocessing": {"fill_from": {"p01m": "precipitation"},
                          # Unscaled copies for the physics path. copy_cols runs AFTER fill_from,
                          # so asos_raw carries the gridded value where the station was missing,
                          # which makes the station innovation exactly zero there; asos_observed
                          # (taken BEFORE the fill) records where the station was genuinely
                          # reporting.
                          "copy_cols": {"sw_raw": "shortwave_radiation",
                                        "precip_raw": "precipitation",
                                        "pet_raw": "pet_mm_hr",
                                        "asos_raw": "p01m"},
                          "observed_mask_cols": {"asos_observed": "p01m"},
                          "lapse": {"source": "temperature", "target": "temp_lapse_k"},
                          "swe_col": "snodas_swe_mm"},
        "basins": basins,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=1)
    n_train = sum(1 for b in basins if b["split"] == "train")
    n_embedded = sum(1 for b in basins if b["has_embedding"])
    print("Wrote %s: %d basins (%d train / %d holdout), %d with pretrained embeddings"
          % (args.out, len(basins), n_train, len(basins) - n_train, n_embedded))


if __name__ == "__main__":
    main()
