"""
Data-coverage census for the multi-basin forecast windows.

Answers the question "what does each candidate validity rule actually cost us?" before any rule is
adopted. Reads the raw per-basin CSVs, reindexes each onto a strict hourly UTC grid (the step the
production loader is missing, which is why nominal 1,056-row windows can span years of real time),
enumerates windows exactly as the loaders do, and reports how many survive each combination of:

* issue-time (t0) flow being a genuine observation,
* horizon target coverage thresholds,
* longest contiguous target gap limits,
* precipitation coverage (which must never be linearly interpolated).

Usage::

    python experiments/catchment_foundation/gap_census.py \
        --manifest experiments/catchment_foundation/manifests/co_manifest.json
"""
import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SPINUP = 720
HORIZON = 336
WINDOW = SPINUP + HORIZON


def hourly_observed(csv_path: str, columns: List[str], start: Optional[str],
                    end: Optional[str]) -> Dict[str, np.ndarray]:
    """
    Loads one basin onto a strict hourly UTC grid and returns per-column observation masks.

    :param csv_path: Path of the basin's hourly CSV.
    :type csv_path: str
    :param columns: Columns to build masks for.
    :type columns: List[str]
    :param start: Inclusive lower date bound, or None.
    :type start: str, optional
    :param end: Exclusive upper date bound, or None.
    :type end: str, optional
    :return: column -> boolean array (True where observed) on the reindexed grid.
    :rtype: Dict[str, np.ndarray]
    """
    header = pd.read_csv(csv_path, nrows=0).columns
    wanted = [c for c in columns if c in header]
    frame = pd.read_csv(csv_path, usecols=["datetime"] + wanted)
    stamps = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
    frame = frame.assign(datetime=stamps).dropna(subset=["datetime"])
    frame = frame.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index < pd.Timestamp(end)]
    if len(frame) < WINDOW:
        return {}
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    # Count rows AFTER the date filter, so the absent-row statistic compares like with like.
    n_rows = len(frame)
    frame = frame.reindex(grid)
    masks = {c: frame[c].notna().to_numpy() if c in frame.columns
             else np.zeros(len(grid), dtype=bool) for c in columns}
    masks["_n_rows_before_reindex"] = n_rows
    masks["_n_rows_after_reindex"] = len(grid)
    return masks


def has_long_gap(observed: np.ndarray, limit: int, lo: int, hi: int, stride: int,
                 n_windows: int) -> np.ndarray:
    """
    Flags windows containing a run of more than ``limit`` consecutive missing steps.

    A window contains such a run iff some position within it starts ``limit + 1`` consecutive
    missing steps, which is evaluated for every window at once via cumulative sums.

    :param observed: Boolean observation mask over the basin's hourly grid.
    :type observed: np.ndarray
    :param limit: Longest tolerated contiguous gap, in hours.
    :type limit: int
    :param lo: Offset of the checked segment from each window start.
    :type lo: int
    :param hi: End offset (exclusive) of the checked segment from each window start.
    :type hi: int
    :param stride: Spacing between window starts.
    :type stride: int
    :param n_windows: Number of windows.
    :type n_windows: int
    :return: Boolean array, True where the window has an over-limit gap.
    :rtype: np.ndarray
    """
    missing = (~observed).astype(np.int32)
    run = np.convolve(missing, np.ones(limit + 1, dtype=np.int32), mode="valid")
    starts_bad = np.concatenate([(run == limit + 1), np.zeros(limit, dtype=bool)])
    cumulative = np.concatenate([[0], np.cumsum(starts_bad)])
    out = np.zeros(n_windows, dtype=bool)
    for w in range(n_windows):
        a, b = w * stride + lo, w * stride + hi - limit
        if b > a:
            out[w] = (cumulative[b] - cumulative[a]) > 0
    return out


def census_split(basins: List[Dict], start: Optional[str], end: Optional[str],
                 stride: int, label: str) -> None:
    """
    Runs and prints the census for one split.

    :param basins: Manifest basin entries for this split.
    :type basins: List[Dict]
    :param start: Inclusive lower date bound, or None.
    :type start: str, optional
    :param end: Exclusive upper date bound, or None.
    :type end: str, optional
    :param stride: Spacing between window starts, in hours.
    :type stride: int
    :param label: Split name for the printed report.
    :type label: str
    :return: None
    :rtype: None
    """
    cols = ["cfs", "precipitation", "p01m"]
    tot = 0
    t0_ok = 0
    cov = {90: 0, 95: 0, 100: 0}
    gap = {3: 0, 6: 0, 12: 0, 24: 0}
    precip_full = 0
    combined = {"strict": 0, "moderate": 0}
    compressed_rows = 0
    total_rows = 0
    for basin in basins:
        masks = hourly_observed(basin["csv_path"], cols, start, end)
        if not masks:
            continue
        flow = masks["cfs"]
        n = (len(flow) - WINDOW) // stride + 1
        if n <= 0:
            continue
        total_rows += masks["_n_rows_after_reindex"]
        compressed_rows += masks["_n_rows_after_reindex"] - masks["_n_rows_before_reindex"]
        starts = np.arange(n) * stride
        tot += n
        # Issue-time flow: last hour of the spin-up segment.
        t0 = flow[starts + SPINUP - 1]
        t0_ok += int(t0.sum())
        # Horizon target coverage.
        cflow = np.concatenate([[0], np.cumsum(flow.astype(np.int32))])
        tgt_obs = cflow[starts + WINDOW] - cflow[starts + SPINUP]
        frac = tgt_obs / HORIZON
        for k in cov:
            cov[k] += int((frac >= k / 100.0).sum())
        gaps = {limit: has_long_gap(flow, limit, SPINUP, WINDOW, stride, n) for limit in gap}
        for limit in gap:
            gap[limit] += int((~gaps[limit]).sum())
        # Precipitation over the whole window (never linearly interpolable).
        pr = masks["precipitation"] | masks["p01m"]
        cpr = np.concatenate([[0], np.cumsum(pr.astype(np.int32))])
        pr_full = (cpr[starts + WINDOW] - cpr[starts]) == WINDOW
        precip_full += int(pr_full.sum())
        combined["strict"] += int((t0 & (frac >= 0.95) & (~gaps[6]) & pr_full).sum())
        combined["moderate"] += int((t0 & (frac >= 0.90) & (~gaps[12]) & pr_full).sum())

    if tot == 0:
        print("%s: no windows" % label)
        return
    pct = lambda x: 100.0 * x / tot  # noqa: E731
    print("\n=== %s: %d windows across %d basins ===" % (label, tot, len(basins)))
    print("  rows absent from the raw CSVs (revealed by hourly reindex): %d of %d (%.1f%%)"
          % (compressed_rows, total_rows, 100.0 * compressed_rows / max(total_rows, 1)))
    print("  t0 flow genuinely observed              : %6.1f%%" % pct(t0_ok))
    for k in sorted(cov):
        print("  horizon target coverage >= %3d%%         : %6.1f%%" % (k, pct(cov[k])))
    for limit in sorted(gap):
        print("  longest target gap <= %2dh               : %6.1f%%" % (limit, pct(gap[limit])))
    print("  precipitation complete (NLDAS or ASOS)  : %6.1f%%" % pct(precip_full))
    print("  SURVIVING strict   (t0 + >=95%% + <=6h + precip) : %6.1f%%  (n=%d)"
          % (pct(combined["strict"]), combined["strict"]))
    print("  SURVIVING moderate (t0 + >=90%% + <=12h + precip): %6.1f%%  (n=%d)"
          % (pct(combined["moderate"]), combined["moderate"]))


def main() -> None:
    """
    Runs the census over the training and both evaluation splits.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    with open(args.manifest) as f:
        manifest = json.load(f)
    train = [b for b in manifest["basins"] if b.get("split") == "train"]
    holdout = [b for b in manifest["basins"] if b.get("split") == "holdout"]
    census_split(train, "2023-01-01", None, 336, "EVAL gauged_2023")
    census_split(holdout, "2023-01-01", None, 336, "EVAL ungauged_2023")
    census_split(train, None, "2022-01-01", 72, "TRAIN (pre-2022)")


if __name__ == "__main__":
    main()
