#!/usr/bin/env python3
"""Compare saved forecast HTMLs visually and numerically.

The evaluator writes Plotly HTML files containing the actual hydrograph and
forecast.  This utility extracts those traces without needing a browser and
builds one compact PNG gallery with a shared visual vocabulary:

* black: observed flow
* gray: pre-issue history
* gray dashed: persistence forecast
* red: reference model
* blue: candidate model

It is intentionally complementary to aggregate metrics.  The case-level JSON
captures common hydrograph failure modes such as muted peaks, timing error,
bias, and negative flow.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PLOTLY_CALL = re.compile(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*')


@dataclass
class TraceMetrics:
    mse: float
    persistence_mse: float
    skill_vs_persistence_pct: float
    correlation: float
    amplitude_ratio: float
    mean_bias: float
    peak_ratio: float
    peak_lag_hours: int
    negative_fraction: float


def _finite_array(values: list[Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )


def _load_plotly_traces(path: Path) -> dict[str, Any]:
    html = path.read_text(encoding="utf-8")
    matches = list(_PLOTLY_CALL.finditer(html))
    if not matches:
        raise ValueError(f"No Plotly.newPlot call found in {path}")

    decoder = json.JSONDecoder()
    traces, _ = decoder.raw_decode(html[matches[-1].end() :])
    by_name = {str(trace.get("name", "")): trace for trace in traces}

    pred_trace = by_name.get("preds")
    if pred_trace is None:
        pred_trace = next(
            (trace for trace in traces if "pred" in str(trace.get("name", "")).lower()),
            None,
        )
    if pred_trace is None:
        raise ValueError(f"No prediction trace found in {path}")

    actual_trace = next(
        (
            trace
            for trace in traces
            if trace is not pred_trace
            and str(trace.get("name", "")).lower()
            not in {"pred_start", "prediction start"}
            and len(trace.get("y", [])) == len(pred_trace.get("y", []))
        ),
        None,
    )
    if actual_trace is None:
        raise ValueError(f"No actual-flow trace found in {path}")

    pred = _finite_array(pred_trace["y"])
    actual = _finite_array(actual_trace["y"])
    finite_pred = np.flatnonzero(np.isfinite(pred))
    if finite_pred.size == 0:
        raise ValueError(f"Prediction trace contains no finite values in {path}")
    issue_index = int(finite_pred[0])

    return {
        "x": pd.to_datetime(pred_trace["x"]),
        "actual": actual,
        "pred": pred,
        "issue_index": issue_index,
    }


def _trace_metrics(actual: np.ndarray, pred: np.ndarray, persistence: float) -> TraceMetrics:
    mask = np.isfinite(actual) & np.isfinite(pred)
    obs = actual[mask]
    sim = pred[mask]
    if not obs.size:
        raise ValueError("No overlapping finite observations and predictions")

    mse = float(np.mean((sim - obs) ** 2))
    persistence_mse = float(np.mean((persistence - obs) ** 2))
    skill = (
        float(100.0 * (1.0 - mse / persistence_mse))
        if persistence_mse > 0
        else float("nan")
    )
    corr = (
        float(np.corrcoef(sim, obs)[0, 1])
        if np.std(sim) > 0 and np.std(obs) > 0
        else float("nan")
    )
    amplitude_ratio = (
        float(np.std(sim) / np.std(obs)) if np.std(obs) > 0 else float("nan")
    )
    observed_peak = float(np.max(obs))
    peak_ratio = float(np.max(sim) / observed_peak) if observed_peak != 0 else float("nan")

    return TraceMetrics(
        mse=mse,
        persistence_mse=persistence_mse,
        skill_vs_persistence_pct=skill,
        correlation=corr,
        amplitude_ratio=amplitude_ratio,
        mean_bias=float(np.mean(sim - obs)),
        peak_ratio=peak_ratio,
        peak_lag_hours=int(np.argmax(sim) - np.argmax(obs)),
        negative_fraction=float(np.mean(sim < 0)),
    )


def _case_record(
    reference_path: Path,
    candidate_path: Path,
    reference_root: Path,
    candidate_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference = _load_plotly_traces(reference_path)
    candidate = _load_plotly_traces(candidate_path)
    if reference["issue_index"] != candidate["issue_index"]:
        raise ValueError(f"Issue index differs for {reference_path} and {candidate_path}")
    if not np.array_equal(reference["x"].values, candidate["x"].values):
        raise ValueError(f"Timestamps differ for {reference_path} and {candidate_path}")

    issue_index = candidate["issue_index"]
    persistence = float(candidate["actual"][issue_index - 1])
    observed = candidate["actual"][issue_index:]
    ref_pred = reference["pred"][issue_index:]
    cand_pred = candidate["pred"][issue_index:]
    ref_metrics = _trace_metrics(observed, ref_pred, persistence)
    cand_metrics = _trace_metrics(observed, cand_pred, persistence)
    relative = candidate_path.relative_to(candidate_root)
    parts = relative.parts
    site_id = parts[-2] if len(parts) >= 2 else "unknown"
    issue_time = candidate["x"][issue_index]

    record = {
        "case": str(relative),
        "site_id": site_id,
        "issue_time": issue_time.isoformat(),
        "observed_peak_cfs": float(np.nanmax(observed)),
        "issue_flow_cfs": persistence,
        "reference": asdict(ref_metrics),
        "candidate": asdict(cand_metrics),
        "candidate_minus_reference_skill_pct": float(
            cand_metrics.skill_vs_persistence_pct
            - ref_metrics.skill_vs_persistence_pct
        ),
        "reference_html": str(reference_path),
        "candidate_html": str(candidate_path),
    }
    return record, reference, candidate


def _select_cases(records: list[dict[str, Any]], maximum: int) -> list[int]:
    selectors = (
        ("observed_peak_cfs", True),
        ("candidate.skill_vs_persistence_pct", False),
        ("candidate.skill_vs_persistence_pct", True),
        ("candidate_minus_reference_skill_pct", True),
    )
    selected: list[int] = []
    per_selector = max(1, math.ceil(maximum / len(selectors)))

    def value(record: dict[str, Any], key: str) -> float:
        current: Any = record
        for part in key.split("."):
            current = current[part]
        result = float(current)
        return result if np.isfinite(result) else -np.inf

    for key, reverse in selectors:
        order = sorted(
            range(len(records)),
            key=lambda index: value(records[index], key),
            reverse=reverse,
        )
        for index in order[:per_selector]:
            if index not in selected:
                selected.append(index)
            if len(selected) >= maximum:
                return selected
    return selected


def _plot_gallery(
    records: list[dict[str, Any]],
    traces: list[tuple[dict[str, Any], dict[str, Any]]],
    selected: list[int],
    output_path: Path,
    reference_label: str,
    candidate_label: str,
) -> None:
    columns = 2
    rows = max(1, math.ceil(len(selected) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(17, 4.7 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for panel, index in enumerate(selected):
        axis = axes.flat[panel]
        record = records[index]
        reference, candidate = traces[index]
        issue_index = candidate["issue_index"]
        x = candidate["x"]
        persistence = record["issue_flow_cfs"]

        axis.plot(
            x[: issue_index + 1],
            candidate["actual"][: issue_index + 1],
            color="#888888",
            linewidth=1.4,
            label="Observed history",
        )
        axis.plot(
            x[issue_index:],
            candidate["actual"][issue_index:],
            color="#111111",
            linewidth=2.1,
            label="Observed future",
        )
        axis.plot(
            x[issue_index:],
            reference["pred"][issue_index:],
            color="#d1495b",
            linewidth=1.5,
            alpha=0.9,
            label=reference_label,
        )
        axis.plot(
            x[issue_index:],
            candidate["pred"][issue_index:],
            color="#0077b6",
            linewidth=1.8,
            label=candidate_label,
        )
        axis.plot(
            x[issue_index:],
            np.full(len(x) - issue_index, persistence),
            color="#777777",
            linewidth=1.2,
            linestyle="--",
            label="Persistence",
        )
        axis.axvline(x[issue_index], color="#555555", linewidth=1.0, linestyle=":")
        candidate_metrics = record["candidate"]
        axis.set_title(
            f"{record['site_id']} · {pd.Timestamp(record['issue_time']).date()}\n"
            f"skill {candidate_metrics['skill_vs_persistence_pct']:+.1f}% · "
            f"corr {candidate_metrics['correlation']:+.2f} · "
            f"amp {candidate_metrics['amplitude_ratio']:.2f}× · "
            f"peak lag {candidate_metrics['peak_lag_hours']:+d} h",
            fontsize=10,
        )
        axis.set_ylabel("Flow (cfs)")
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", rotation=25)

    for panel in range(len(selected), rows * columns):
        axes.flat[panel].axis("off")

    axes.flat[0].legend(loc="best", fontsize=8, frameon=False)
    figure.suptitle(
        "Forecast shape diagnostics — high peaks, worst/best skill, largest improvement",
        fontsize=15,
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--split", default="eval_gauged_2023")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-label", default="Reference")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--max-panels", type=int, default=8)
    args = parser.parse_args()

    reference_root = args.reference_dir / args.split
    candidate_root = args.candidate_dir / args.split
    relative_paths = sorted(
        set(path.relative_to(reference_root) for path in reference_root.rglob("forecast_*.html"))
        & set(path.relative_to(candidate_root) for path in candidate_root.rglob("forecast_*.html"))
    )
    if not relative_paths:
        raise SystemExit(
            f"No matching forecast HTML files under {reference_root} and {candidate_root}"
        )

    records: list[dict[str, Any]] = []
    traces: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for relative in relative_paths:
        record, reference, candidate = _case_record(
            reference_root / relative,
            candidate_root / relative,
            reference_root,
            candidate_root,
        )
        records.append(record)
        traces.append((reference, candidate))

    selected = _select_cases(records, max(1, args.max_panels))
    output_dir = args.output_dir or args.candidate_dir / f"visual_compare_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "case_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "reference_dir": str(args.reference_dir),
                "candidate_dir": str(args.candidate_dir),
                "split": args.split,
                "selected_cases": [records[index]["case"] for index in selected],
                "cases": records,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    gallery_path = output_dir / "forecast_gallery.png"
    _plot_gallery(
        records,
        traces,
        selected,
        gallery_path,
        args.reference_label,
        args.candidate_label,
    )
    print(f"Wrote {gallery_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
