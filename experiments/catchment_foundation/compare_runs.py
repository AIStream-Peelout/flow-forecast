"""
Compare two catchment-foundation run directories on their saved evaluation artifacts.

The comparison refuses to silently treat different evaluation cohorts as equivalent: it reports
site-set differences and checks the persistence MSE, which should be identical when both models
were evaluated on the same windows.

Example::

    python experiments/catchment_foundation/compare_runs.py \
        experiments/catchment_foundation/runs/fleet_hybrid \
        experiments/catchment_foundation/runs/crossformer_hindcast_v1
"""
import argparse
import json
import os
from typing import Dict

import numpy as np

SPLITS = ("gauged_2023", "ungauged_2023")
BANDS = ("day1-3", "day4-7", "day8-14", "all")


def _read_json(path: str) -> Dict:
    """Reads one required JSON artifact with a useful missing-file error."""
    if not os.path.exists(path):
        raise FileNotFoundError("Missing evaluation artifact: %s" % path)
    with open(path) as f:
        return json.load(f)


def compare_run_directories(reference_dir: str, candidate_dir: str) -> Dict:
    """
    Compares pooled and per-basin skill for two completed evaluation runs.

    Positive deltas and win rates above 50% favour the candidate. ``same_persistence_mse`` is the
    key cohort check: false means the models did not score the same observations/windows, so the
    skill delta is not a controlled model comparison.

    :param reference_dir: Run directory for the reference (for example the hybrid model).
    :type reference_dir: str
    :param candidate_dir: Run directory for the candidate (for example direct Crossformer).
    :type candidate_dir: str
    :return: Structured split/band comparison.
    :rtype: Dict
    """
    report = {
        "reference_dir": os.path.abspath(reference_dir),
        "candidate_dir": os.path.abspath(candidate_dir),
        "splits": {},
    }
    for split in SPLITS:
        reference_eval = os.path.join(reference_dir, "eval_" + split)
        candidate_eval = os.path.join(candidate_dir, "eval_" + split)
        reference_pooled = _read_json(os.path.join(reference_eval, "pooled_metrics.json"))
        candidate_pooled = _read_json(os.path.join(candidate_eval, "pooled_metrics.json"))
        reference_basins = _read_json(os.path.join(reference_eval, "per_basin_metrics.json"))
        candidate_basins = _read_json(os.path.join(candidate_eval, "per_basin_metrics.json"))
        reference_sites = set(reference_basins)
        candidate_sites = set(candidate_basins)
        common_sites = sorted(reference_sites & candidate_sites)
        split_report = {
            "same_site_set": reference_sites == candidate_sites,
            "reference_only_sites": sorted(reference_sites - candidate_sites),
            "candidate_only_sites": sorted(candidate_sites - reference_sites),
            "n_common_basins": len(common_sites),
            "bands": {},
        }
        for band in BANDS:
            ref_persistence = reference_pooled[band]["mse_persistence_mm_hr2"]
            cand_persistence = candidate_pooled[band]["mse_persistence_mm_hr2"]
            same_persistence = bool(np.isclose(ref_persistence, cand_persistence,
                                               rtol=1e-6, atol=1e-10))
            ref_skills = np.array([
                reference_basins[site][band]["skill_vs_persistence_pct"]
                for site in common_sites], dtype=float)
            cand_skills = np.array([
                candidate_basins[site][band]["skill_vs_persistence_pct"]
                for site in common_sites], dtype=float)
            deltas = cand_skills - ref_skills
            split_report["bands"][band] = {
                "same_persistence_mse": same_persistence,
                "reference_persistence_mse_mm_hr2": ref_persistence,
                "candidate_persistence_mse_mm_hr2": cand_persistence,
                "reference_pooled_skill_pct":
                    reference_pooled[band]["skill_vs_persistence_pct"],
                "candidate_pooled_skill_pct":
                    candidate_pooled[band]["skill_vs_persistence_pct"],
                "pooled_skill_delta_pct_points": round(
                    candidate_pooled[band]["skill_vs_persistence_pct"]
                    - reference_pooled[band]["skill_vs_persistence_pct"], 3),
                "reference_median_basin_skill_pct":
                    round(float(np.median(ref_skills)), 3) if len(ref_skills) else None,
                "candidate_median_basin_skill_pct":
                    round(float(np.median(cand_skills)), 3) if len(cand_skills) else None,
                "median_basin_skill_delta_pct_points":
                    round(float(np.median(deltas)), 3) if len(deltas) else None,
                "candidate_basin_win_rate_pct":
                    round(100.0 * float(np.mean(deltas > 0)), 1) if len(deltas) else None,
            }
        report["splits"][split] = split_report
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = compare_run_directories(args.reference_dir, args.candidate_dir)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        with open(args.out, "w") as f:
            f.write(rendered + "\n")


if __name__ == "__main__":
    main()
