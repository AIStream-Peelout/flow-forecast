"""
Multi-catchment forecast training through flow-forecast's native config pipeline.

Builds the FF config for ``HybridGR4MultiBasin`` + ``MultiBasinCatchmentWindow``, trains with
``train_transformer_style`` (W&B logging via FF), then evaluates both holdout splits
(gauged-time-holdout 2023+, ungauged-basin 2023+) with ``forecast_report``.

Usage (smoke)::

    python experiments/catchment_foundation/run_training.py --name smoke10 --max-basins 10 \
        --epochs 3 --samples-per-epoch 512

Usage (full fleet)::

    python experiments/catchment_foundation/run_training.py --name fleet_v1 --epochs 20 \
        --samples-per-epoch 4096
"""
import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

MANIFEST_DEFAULT = os.path.join(os.path.dirname(__file__), "manifests", "co_manifest.json")
RELEVANT_COLS = ["cfs", "precipitation", "temperature", "shortwave_radiation",
                 "longwave_radiation", "specific_humidity", "wind_east", "wind_north", "p01m",
                 "pet_mm_hr", "temp_lapse_k", "sw_raw"]
SCALED_COLS = RELEVANT_COLS[1:10]
SPINUP_HOURS = 720
HORIZON_HOURS = 336
TRAIN_END = "2022-01-01"
TEST_START = "2023-01-01"


def load_env(path: str) -> None:
    """
    Loads KEY=VALUE lines from a .env file into the environment (no override).

    :param path: The .env file path.
    :type path: str
    :return: None
    :rtype: None
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def build_params(manifest_path: str, run_name: str, epochs: int, batch_size: int,
                 samples_per_epoch: int, max_basins: Optional[int], lr: float,
                 use_wandb: bool, use_swe: bool = False) -> Dict:
    """
    Builds the FF config dict for the multi-basin run.

    :param manifest_path: The basin manifest JSON path.
    :type manifest_path: str
    :param run_name: The W&B run name.
    :type run_name: str
    :param epochs: Training epochs.
    :type epochs: int
    :param batch_size: Windows per batch.
    :type batch_size: int
    :param samples_per_epoch: Weighted-sampler draws per epoch.
    :type samples_per_epoch: int
    :param max_basins: Optional basin cap (smoke runs), defaults to None.
    :type max_basins: Optional[int]
    :param lr: Adam learning rate.
    :type lr: float
    :param use_wandb: Whether to enable W&B logging.
    :type use_wandb: bool
    :param use_swe: Whether to add the SNODAS SWE channel and seed the snow store from it
        (requires swe series referenced by the manifest), defaults to False.
    :type use_swe: bool, optional
    :return: The config dict.
    :rtype: Dict
    """
    relevant_cols = RELEVANT_COLS + (["snodas_swe_mm"] if use_swe else [])
    params = {
        "model_name": "HybridGR4MultiBasin",
        "model_type": "PyTorch",
        "model_params": {
            "n_time_series": len(relevant_cols) + 1,
            "spinup_length": SPINUP_HOURS,
            "forecast_length": HORIZON_HOURS,
            "raw_temp_index": relevant_cols.index("temp_lapse_k"),
            "raw_sw_index": relevant_cols.index("sw_raw"),
            "basin_info_path": manifest_path,
            "context_dim": 256, "dim": 64, "depth": 2, "heads": 4, "snow": True,
        },
        "dataset_params": {
            "class": "MultiBasinCatchmentWindow",
            "training_path": manifest_path, "validation_path": manifest_path,
            "test_path": manifest_path,
            "batch_size": batch_size,
            "forecast_history": SPINUP_HOURS, "forecast_length": HORIZON_HOURS,
            "target_col": ["cfs"], "relevant_cols": relevant_cols, "scaled_cols": SCALED_COLS,
            "window_stride": 72, "min_valid_fraction": 0.95,
            "train_basin_split": "train", "valid_basin_split": "train",
            "test_basin_split": "train",
            "train_end_date": TRAIN_END,
            "valid_start_date": TRAIN_END, "valid_end_date": TEST_START,
            "test_start_date": TEST_START,
            "valid_window_stride": 336, "test_window_stride": 672,
            "train_samples_per_epoch": samples_per_epoch,
        },
        "early_stopping": {"patience": 3},
        "training_params": {"criterion": "MSE", "optimizer": "Adam", "optim_params": {},
                            "lr": lr, "epochs": epochs, "batch_size": batch_size},
        "GCS": False,
        "wandb": {"project": "catchment-foundation", "name": run_name,
                  "tags": ["multi_basin", "hybrid_gr4"]} if use_wandb else False,
        "forward_params": {},
        "metrics": ["MSE"],
    }
    if use_swe:
        params["model_params"]["swe_index"] = relevant_cols.index("snodas_swe_mm")
        if params["wandb"]:
            params["wandb"]["tags"].append("snodas_swe")
    if max_basins is not None:
        params["dataset_params"]["max_basins"] = max_basins
    return params


def main() -> None:
    """
    Trains and evaluates the multi-basin forecaster.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Run name (W&B and output dir)")
    parser.add_argument("--manifest", default=MANIFEST_DEFAULT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--max-basins", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--swe", action="store_true",
                        help="Add the SNODAS SWE channel and seed the snow store from it")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--eval-stride", type=int, default=336,
                        help="Window stride (hours) for the post-training forecast_report eval")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    load_env(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    os.environ.setdefault("WANDB_ENTITY", "igodfried")

    run_dir = os.path.join(os.path.dirname(__file__), "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    params = build_params(args.manifest, args.name, args.epochs, args.batch_size,
                          args.samples_per_epoch, args.max_basins, args.lr, not args.no_wandb,
                          use_swe=args.swe)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(params, f, indent=1)

    from flood_forecast.pytorch_training import train_transformer_style
    from flood_forecast.time_model import PyTorchForecast

    model = PyTorchForecast(params["model_name"], args.manifest, args.manifest, args.manifest,
                            params)
    print("train windows: %d (%d basins), valid windows: %d, test windows: %d"
          % (len(model.training), len(model.training.basin_loaders), len(model.validation),
             len(model.test_data)))
    train_transformer_style(model, params["training_params"], forward_params={},
                            model_filepath=run_dir)
    print("Training complete; model saved under %s" % run_dir)

    if not args.skip_eval:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from evaluate import evaluate_splits
        evaluate_splits(model, args.manifest, run_dir, eval_stride=args.eval_stride,
                        max_basins=args.max_basins)


if __name__ == "__main__":
    main()
