"""
Split-wise evaluation of a trained multi-basin forecaster with ``forecast_report``.

For each requested split — "gauged_2023" (basins seen in training),
"basin_valid_2023" (development basins held out of hypernetwork fitting), and
"ungauged_2023" (final basin holdout) — runs the model over every eval window, converts
simulated/observed/persistence flows back to physical mm/hr, and produces the standard
:func:`~flood_forecast.ode.physics.forecast_training.forecast_report` per basin (metrics JSON +
example forecast plots on disk). Pooled and per-basin metrics are logged to W&B; the headline
number is skill vs persistence at day 1-3.
"""
import json
import math
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

BANDS = {"day1-3": (0, 72), "day4-7": (72, 168), "day8-14": (168, 336), "all": (0, 336)}


def _window_shape_metrics(sim: np.ndarray, obs: np.ndarray,
                          persist: np.ndarray) -> Dict[str, float]:
    """Returns hydrograph-shape diagnostics for one forecast window."""
    mse = float(np.mean((sim - obs) ** 2))
    persistence_mse = float(np.mean((persist - obs) ** 2))
    skill = (100.0 * (1.0 - mse / persistence_mse)
             if persistence_mse > 0 else float("nan"))
    correlation = (float(np.corrcoef(sim, obs)[0, 1])
                   if np.std(sim) > 0 and np.std(obs) > 0 else float("nan"))
    amplitude_ratio = (float(np.std(sim) / np.std(obs))
                       if np.std(obs) > 0 else float("nan"))
    observed_peak = float(np.max(obs))
    return {
        "mse_mm_hr2": mse,
        "persistence_mse_mm_hr2": persistence_mse,
        "skill_vs_persistence_pct": skill,
        "correlation": correlation,
        "amplitude_ratio": amplitude_ratio,
        "mean_bias_mm_hr": float(np.mean(sim - obs)),
        "peak_ratio": (float(np.max(sim) / observed_peak)
                       if observed_peak != 0 else float("nan")),
        "peak_lag_hours": int(np.argmax(sim) - np.argmax(obs)),
        "negative_fraction": float(np.mean(sim < 0)),
    }


def save_forecast_gallery(split_outputs: Dict[str, Dict], loader, areas: Dict[str, float],
                          out_dir: str, split_name: str, max_panels: int = 8) -> str:
    """
    Saves a compact PNG gallery chosen to expose hydrograph failure modes.

    Cases are selected across the complete split by observed peak, best/worst skill versus
    persistence, amplitude blow-up and absolute peak-timing error.  This makes forecast shape
    review a standard evaluation artifact instead of requiring users to open many HTML files.

    :return: Absolute path to the PNG gallery.
    :rtype: str
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    site_to_local = {site: i for i, site in enumerate(loader.basin_site_ids)}
    cases = []
    for site, data in split_outputs.items():
        sim = data["sim"].numpy()
        obs = data["obs"].numpy()
        persist = data["persist"].numpy()
        for index in range(sim.shape[0]):
            record = {
                "site_id": site,
                "window_index": index,
                "issue_position": int(data["t0s"][index]),
                "observed_peak_mm_hr": float(np.max(obs[index])),
                "metrics": _window_shape_metrics(sim[index], obs[index], persist[index]),
            }
            cases.append(record)

    def metric(record, key):
        value = (record[key] if key in record else record["metrics"][key])
        value = float(value)
        return value if np.isfinite(value) else -np.inf

    selectors = [
        ("observed_peak_mm_hr", True, 2),
        ("skill_vs_persistence_pct", False, 2),
        ("skill_vs_persistence_pct", True, 2),
        ("amplitude_ratio", True, 1),
    ]
    selected = []
    for key, reverse, count in selectors:
        ordered = sorted(range(len(cases)), key=lambda i: metric(cases[i], key),
                         reverse=reverse)
        for index in ordered[:count]:
            if index not in selected:
                selected.append(index)
    timing_order = sorted(
        range(len(cases)),
        key=lambda i: abs(cases[i]["metrics"]["peak_lag_hours"]),
        reverse=True)
    for index in timing_order:
        if index not in selected:
            selected.append(index)
            break
    selected = selected[:max_panels]

    columns = 2
    rows = max(1, math.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(17, 4.5 * rows), squeeze=False)
    history_hours = 72
    selected_records = []
    for panel, case_index in enumerate(selected):
        axis = axes.flat[panel]
        record = cases[case_index]
        site = record["site_id"]
        local = site_to_local[site]
        data = split_outputs[site]
        window_index = record["window_index"]
        issue_position = record["issue_position"]
        timestamps = loader.basin_timestamps[local]
        history_start = max(0, issue_position - history_hours)
        horizon = data["sim"].shape[1]
        history = loader.basin_loaders[local].df[
            loader.target_col_list[0]].to_numpy()[history_start:issue_position]
        to_cfs = areas[site] / (0.0283168 * 3.6)
        sim = data["sim"][window_index].numpy() * to_cfs
        obs = data["obs"][window_index].numpy() * to_cfs
        persist = data["persist"][window_index].numpy() * to_cfs
        issue_time = timestamps[issue_position]

        axis.plot(timestamps[history_start:issue_position], history * to_cfs,
                  color="#888888", linewidth=1.3, label="Observed history")
        axis.plot(timestamps[issue_position:issue_position + horizon], obs,
                  color="#111111", linewidth=2.0, label="Observed future")
        axis.plot(timestamps[issue_position:issue_position + horizon], sim,
                  color="#0077b6", linewidth=1.7, label="Model")
        axis.plot(timestamps[issue_position:issue_position + horizon], persist,
                  color="#777777", linewidth=1.2, linestyle="--", label="Persistence")
        axis.axvline(issue_time, color="#555555", linewidth=1.0, linestyle=":")
        metrics = record["metrics"]
        axis.set_title(
            "%s · %s\nskill %+.1f%% · corr %+.2f · amp %.2f× · peak lag %+d h"
            % (site, str(issue_time)[:10], metrics["skill_vs_persistence_pct"],
               metrics["correlation"], metrics["amplitude_ratio"],
               metrics["peak_lag_hours"]),
            fontsize=10)
        axis.set_ylabel("Flow (cfs)")
        axis.grid(alpha=0.2)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        if panel == 0:
            axis.legend(loc="best", fontsize=8, frameon=False)
        selected_records.append({
            **record,
            "issue_time": issue_time.isoformat(),
            "observed_peak_cfs": record["observed_peak_mm_hr"] * to_cfs,
        })

    for panel in range(len(selected), rows * columns):
        axes.flat[panel].axis("off")
    figure.suptitle(
        "%s forecast diagnostics — peaks, worst/best skill, amplitude and timing failures"
        % split_name,
        fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.975])
    os.makedirs(out_dir, exist_ok=True)
    gallery_path = os.path.abspath(os.path.join(out_dir, "forecast_gallery.png"))
    figure.savefig(gallery_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    with open(os.path.join(out_dir, "forecast_gallery_cases.json"), "w") as file:
        json.dump({
            "selection": ("two largest peaks, two worst and two best persistence-relative "
                          "windows, largest amplitude ratio, largest absolute peak lag"),
            "selected_cases": selected_records,
            "all_cases": cases,
        }, file, indent=2, allow_nan=True)
    return gallery_path


def collect_split_outputs(model, loader) -> Dict[str, Dict]:
    """
    Runs the model over every window of a split loader and groups physical-unit outputs by basin.

    :param model: A trained multi-basin forecaster exposing per-manifest ``flow_scales``.
    :type model: torch.nn.Module
    :param loader: A MultiBasinWindowLoader for the split.
    :type loader: MultiBasinWindowLoader
    :return: site_id -> {"sim", "obs", "persist" (n_windows, horizon) mm/hr tensors, "t0s"}.
    :rtype: Dict[str, Dict]
    """
    from torch.utils.data import DataLoader
    model.eval()
    device = next(model.parameters()).device
    spinup = loader.forecast_history
    outputs = {site: {"sim": [], "obs": [], "persist": [], "t0s": []}
               for site in loader.basin_site_ids}
    position_to_local = {pos: i for i, pos in enumerate(loader.basin_positions)}
    batch_iter = DataLoader(loader, batch_size=16, shuffle=False)
    window = 0
    with torch.no_grad():
        for src, trg in batch_iter:
            sim_std = model(src.to(device))
            positions = src[:, 0, -1].long()
            scales = model.flow_scales[positions.to(model.flow_scales.device)].unsqueeze(1)
            sim_mm = (sim_std * scales).cpu()
            obs_mm = trg[:, :, 0] * scales.cpu()
            persist_mm = src[:, spinup - 1, 0:1].expand(-1, sim_mm.shape[1]).cpu()
            for row in range(src.shape[0]):
                local = position_to_local[int(positions[row])]
                site = loader.basin_site_ids[local]
                outputs[site]["sim"].append(sim_mm[row])
                outputs[site]["obs"].append(obs_mm[row])
                outputs[site]["persist"].append(persist_mm[row])
                _, local_idx = loader.locate(window)
                start = loader.basin_loaders[local].valid_starts[local_idx]
                outputs[site]["t0s"].append(start + spinup)
                window += 1
    return {site: {key: (torch.stack(value) if key != "t0s" else torch.tensor(value))
                   for key, value in data.items()}
            for site, data in outputs.items() if data["sim"]}


def pooled_metrics(split_outputs: Dict[str, Dict]) -> Dict:
    """
    Computes pooled per-band metrics over all windows of a split (forecast_report formulas).

    :param split_outputs: The per-basin outputs from :func:`collect_split_outputs`.
    :type split_outputs: Dict[str, Dict]
    :return: Band -> {mse, persistence mse, skill %, median window NSE}.
    :rtype: Dict
    """
    sim = torch.cat([d["sim"] for d in split_outputs.values()]).numpy()
    obs = torch.cat([d["obs"] for d in split_outputs.values()]).numpy()
    persist = torch.cat([d["persist"] for d in split_outputs.values()]).numpy()
    metrics = {}
    for name, (a, b) in BANDS.items():
        mse_model = float(((sim[:, a:b] - obs[:, a:b]) ** 2).mean())
        mse_persist = float(((persist[:, a:b] - obs[:, a:b]) ** 2).mean())
        num = ((sim[:, a:b] - obs[:, a:b]) ** 2).sum(1)
        den = np.maximum(((obs[:, a:b] - obs[:, a:b].mean(1, keepdims=True)) ** 2).sum(1), 1e-8)
        metrics[name] = {
            "mse_mm_hr2": round(mse_model, 8),
            "mse_persistence_mm_hr2": round(mse_persist, 8),
            "skill_vs_persistence_pct": round(100.0 * (1.0 - mse_model / mse_persist), 1),
            "median_window_nse": round(float(np.median(1 - num / den)), 3),
            "n_windows": int(sim.shape[0]),
        }
    return metrics


def evaluate_splits(ff_model, manifest_path: str, run_dir: str, eval_stride: int = 336,
                    max_basins: Optional[int] = None, n_report_basins: int = 3,
                    split_names=None) -> Dict:
    """
    Evaluates requested time/basin holdout splits and logs everything to W&B.

    :param ff_model: The trained PyTorchForecast wrapper (or any object with .model and .params).
    :type ff_model: PyTorchForecast
    :param manifest_path: The basin manifest JSON path.
    :type manifest_path: str
    :param run_dir: Directory for eval outputs.
    :type run_dir: str
    :param eval_stride: Hours between forecast issue times, defaults to 336.
    :type eval_stride: int, optional
    :param max_basins: Optional basin cap matching a smoke training run, defaults to None.
    :type max_basins: int, optional
    :param n_report_basins: Number of basins per split to render full forecast_report plots for,
        defaults to 3.
    :type n_report_basins: int, optional
    :return: split -> pooled metrics.
    :rtype: Dict
    """
    import wandb
    from flood_forecast.ode.physics.forecast_training import forecast_report
    from flood_forecast.preprocessing.pytorch_loaders import MultiBasinWindowLoader

    dataset_params = ff_model.params["dataset_params"]
    manifest = json.load(open(manifest_path))
    areas = {b["site_id"]: b["area_sq_km"] for b in manifest["basins"]}
    wandb_run = wandb.run if ff_model.wandb else None
    splits = {"gauged_2023": "train", "basin_valid_2023": "basin_valid",
              "ungauged_2023": "holdout"}
    available_basin_splits = {basin.get("split") for basin in manifest["basins"]}
    splits = {name: basin_split for name, basin_split in splits.items()
              if basin_split in available_basin_splits}
    if split_names is not None:
        splits = {name: split for name, split in splits.items() if name in split_names}
    all_metrics = {}
    for split_name, basin_split in splits.items():
        loader = MultiBasinWindowLoader(
            manifest_path, dataset_params["forecast_history"],
            dataset_params["forecast_length"], dataset_params["target_col"],
            dataset_params["relevant_cols"], scaled_cols=dataset_params.get("scaled_cols"),
            start_date="2023-01-01", basin_split=basin_split, window_stride=eval_stride,
            min_valid_fraction=dataset_params.get("min_valid_fraction", 0.95),
            max_basins=max_basins if basin_split == "train" else None,
            require_pretrained_embedding=dataset_params.get(
                "require_pretrained_embedding", False))
        print("[%s] %d basins, %d windows" % (split_name, len(loader.basin_loaders),
                                              len(loader)))
        outputs = collect_split_outputs(ff_model.model, loader)
        pooled = pooled_metrics(outputs)
        all_metrics[split_name] = pooled
        out_dir = os.path.join(run_dir, "eval_" + split_name)
        os.makedirs(out_dir, exist_ok=True)

        per_basin = {}
        site_to_local = {site: i for i, site in enumerate(loader.basin_site_ids)}
        # Full forecast_report (plots + JSON) for the largest-variance example basins.
        report_sites = sorted(outputs, key=lambda s: -float(outputs[s]["obs"].var()))
        for rank, site in enumerate(report_sites):
            data = outputs[site]
            basin_dir = os.path.join(out_dir, site) if rank < n_report_basins else None
            local = site_to_local[site]
            flow_col = dataset_params["target_col"][0]
            full_flow = loader.basin_loaders[local].df[flow_col].to_numpy()
            per_basin[site] = forecast_report(
                data["sim"], data["obs"], data["persist"], loader.basin_timestamps[local],
                data["t0s"], full_flow=full_flow, area_km2=areas[site], out_dir=basin_dir,
                wandb_run=None, n_examples=4)
            if basin_dir is not None and wandb_run is not None:
                # Predicted-vs-actual example windows, browsable in the W&B run under
                # <split>/<site>/forecast_NN_<issue date>.
                for name in sorted(os.listdir(basin_dir)):
                    if name.endswith(".html"):
                        key = "%s/%s/%s" % (split_name, site, name[:-5])
                        wandb_run.log({key: wandb.Html(open(os.path.join(basin_dir, name)))})
        with open(os.path.join(out_dir, "per_basin_metrics.json"), "w") as f:
            json.dump(per_basin, f, indent=1)
        with open(os.path.join(out_dir, "pooled_metrics.json"), "w") as f:
            json.dump(pooled, f, indent=1)
        gallery_path = save_forecast_gallery(
            outputs, loader, areas, out_dir, split_name)

        skills = [per_basin[s]["day1-3"]["skill_vs_persistence_pct"] for s in per_basin]
        summary = {
            split_name + "/pooled_day1-3_skill_pct": pooled["day1-3"]["skill_vs_persistence_pct"],
            split_name + "/pooled_all_skill_pct": pooled["all"]["skill_vs_persistence_pct"],
            split_name + "/median_basin_day1-3_skill_pct": float(np.median(skills)),
            split_name + "/pct_basins_positive_day1-3_skill":
                round(100.0 * float(np.mean(np.array(skills) > 0)), 1),
            split_name + "/n_basins": len(per_basin),
        }
        print(json.dumps({split_name: pooled}, indent=1))
        print(json.dumps(summary, indent=1))
        if wandb_run is not None:
            flat = {split_name + "/" + band + "/" + key: value
                    for band, entry in pooled.items() for key, value in entry.items()}
            wandb_run.log(flat)
            wandb_run.log(summary)
            caption = (
                "Actual (black), model (blue), persistence (gray dashed); selected by "
                "peak, skill, amplitude and timing failure modes.")
            wandb_run.log({
                split_name + "/forecast_gallery":
                    wandb.Image(gallery_path, caption=caption),
                # Flat root-level alias so the diagnostic is easy to find in W&B Media without
                # expanding nested key groups.
                "hydrograph_gallery_" + split_name:
                    wandb.Image(gallery_path, caption=caption),
            })
            table = wandb.Table(columns=["site_id", "band", "skill_vs_persistence_pct",
                                         "median_window_nse", "rmse_cfs"])
            for site, bands in per_basin.items():
                for band, entry in bands.items():
                    table.add_data(site, band, entry["skill_vs_persistence_pct"],
                                   entry["median_window_nse"], entry.get("rmse_cfs"))
            wandb_run.log({split_name + "/per_basin_metrics": table})
    return all_metrics
