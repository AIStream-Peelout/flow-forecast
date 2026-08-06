"""
Linear probes from catchment embeddings to hydrologic signatures.

Hydrologic signatures are statistics of the observed flow regime (flashiness, baseflow
fraction, flow-duration slope, seasonality). Unlike calibrated model parameters they carry no
equifinality noise, which makes them the cleanest test of what a catchment representation has
learned. This script computes scale-free signatures from each gauge's training-period hourly
flow, then reports cross-validated ridge-regression R-squared per signature from L2-normalized
embeddings — the headline table of the gauge-representation-learning study.

Signatures are computed on pre-2022 data only. Note the known caveat: the embedding history
tower currently sees post-2022 flow, so probe numbers are an upper bound until banks are
rebuilt with a history cutoff.
"""
import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

TRAIN_END = "2022-01-01"


def flow_signatures(flow: pd.Series) -> Optional[Dict[str, float]]:
    """
    Computes scale-free hydrologic signatures from an hourly flow series.

    :param flow: Hourly flow (any consistent unit) with a datetime index; NaNs allowed.
    :type flow: pd.Series
    :return: Signature name -> value, or None if fewer than two years of valid data.
    :rtype: Dict[str, float], optional
    """
    valid = flow.dropna()
    if len(valid) < 2 * 8760 or float(valid.mean()) <= 0:
        return None
    mean = float(valid.mean())
    daily = valid.resample("1D").mean().dropna()
    # Richards-Baker flashiness: pathlength of daily changes divided by total flow.
    rb_index = float(daily.diff().abs().sum() / daily.sum())
    # Baseflow index via a 30-day rolling minimum, a robust filter-free approximation.
    baseflow = daily.rolling(30, center=True, min_periods=10).min()
    bfi = float((baseflow / daily).clip(0, 1).mean())
    quantiles = valid.quantile([0.05, 0.33, 0.5, 0.66, 0.95])
    epsilon = 1e-6 * mean
    fdc_slope = float(np.log(quantiles[0.33] + epsilon) - np.log(quantiles[0.66] + epsilon))
    high_low_ratio = float(np.log((quantiles[0.95] + epsilon) / (quantiles[0.05] + epsilon)))
    monthly = valid.groupby(valid.index.month).mean()
    melt_fraction = float(monthly.reindex([4, 5, 6, 7]).sum() / monthly.sum())
    cv = float(valid.std() / mean)
    hour_of_day = valid.groupby(valid.index.hour).mean()
    diurnal_strength = float((hour_of_day.max() - hour_of_day.min()) / mean)
    return {"log_mean_flow": float(np.log(mean)), "cv": cv, "rb_flashiness": rb_index,
            "bfi": bfi, "fdc_slope": fdc_slope, "high_low_ratio": high_low_ratio,
            "melt_fraction": melt_fraction, "diurnal_strength": diurnal_strength}


def load_flow(site: str, scrape_roots: Dict[str, str]) -> Optional[pd.Series]:
    """
    Loads a gauge's pre-training-cutoff hourly flow from the state scrape directories.

    :param site: The USGS site id.
    :type site: str
    :param scrape_roots: State -> scrape directory mapping.
    :type scrape_roots: Dict[str, str]
    :return: The hourly cfs series before TRAIN_END, or None if the record is absent.
    :rtype: pd.Series, optional
    """
    for root in scrape_roots.values():
        path = os.path.join(root, site, "%s_hourly_full.csv" % site)
        if os.path.exists(path):
            frame = pd.read_csv(path, usecols=["datetime", "cfs"], parse_dates=["datetime"])
            frame = frame.set_index("datetime").sort_index()
            if frame.index.tz is not None:
                frame.index = frame.index.tz_convert("UTC").tz_localize(None)
            flow = pd.to_numeric(frame["cfs"], errors="coerce")
            flow = flow[flow.index < pd.Timestamp(TRAIN_END)]
            return flow.resample("1h").mean()
    return None


def main() -> None:
    """
    CLI entry point: computes signatures and cross-validated probe R-squared per signature.

    :return: None
    :rtype: None
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    parser = argparse.ArgumentParser(description="Signature probes of catchment embeddings")
    parser.add_argument("--embedding-bank", required=True)
    parser.add_argument("--scrape-root", default="/Users/isaac/Documents/GitHub/Water/"
                                                 "pilot_data/scrapes")
    parser.add_argument("--states", nargs="+", default=["CO", "UT"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bank = torch.load(args.embedding_bank, weights_only=False)
    embeddings = torch.nn.functional.normalize(bank["embeddings"], dim=-1).numpy()
    scrape_roots = {s: os.path.join(args.scrape_root, s) for s in args.states}

    rows, kept_embeddings = [], []
    for index, site in enumerate(bank["site_ids"]):
        flow = load_flow(site, scrape_roots)
        signatures = flow_signatures(flow) if flow is not None else None
        if signatures is not None:
            rows.append({"site_id": site, **signatures})
            kept_embeddings.append(embeddings[index])
    table = pd.DataFrame(rows).set_index("site_id")
    features = np.stack(kept_embeddings)
    print("probing %d sites with %d-dim embeddings" % (len(table), features.shape[1]))

    folds = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    probe_r2 = {}
    for signature in table.columns:
        target = table[signature].to_numpy()
        predictions = np.zeros_like(target)
        for train_index, test_index in folds.split(features):
            model = Ridge(alpha=args.alpha)
            model.fit(features[train_index], target[train_index])
            predictions[test_index] = model.predict(features[test_index])
        residual = float(((predictions - target) ** 2).sum())
        variance = float(((target - target.mean()) ** 2).sum())
        probe_r2[signature] = round(1.0 - residual / max(variance, 1e-12), 3)
    report = {"n_sites": len(table), "alpha": args.alpha, "probe_r2": probe_r2,
              "signature_medians": {c: round(float(table[c].median()), 3)
                                    for c in table.columns}}
    table.to_csv(args.output.replace(".json", "_signatures.csv"))
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["probe_r2"], indent=2))


if __name__ == "__main__":
    main()
