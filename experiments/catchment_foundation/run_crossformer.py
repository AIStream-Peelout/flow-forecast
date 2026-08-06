"""
Train the direct Crossformer control for the catchment foundation-model experiment.

The script deliberately reuses the hybrid experiment's manifest, multi-basin loader, temporal
splits, trainer, evaluator, persistence baseline and W&B project. Direct-model training samples
valid windows uniformly by default; ``--event-sample-power 1`` restores the event-variance
weighting used by the original hybrid experiment.
``CrossformerMultiBasin`` directly predicts standardized flow as a residual around persistence,
with no ODE or rainfall-runoff parameterization. Its input block is neural-only: ODE duplicates
are omitted and standardized outliers are clipped by default (disable with ``--input-clip 0``).

By default this is an apples-to-apples *hindcast*: it sees the same realized horizon meteorology
as the hybrid. Pass ``--history-only`` for the operational control that sees only the 30-day
history. The two modes must not be compared as if they solve the same information problem.

Examples::

    # Fast pipeline and overfit sanity check.
    python experiments/catchment_foundation/run_crossformer.py \
        --name crossformer_smoke --max-basins 3 --epochs 3 --samples-per-epoch 128 --no-wandb

    # Fleet-scale apples-to-apples control, monitored in W&B.
    python experiments/catchment_foundation/run_crossformer.py \
        --name crossformer_hindcast_v1 --epochs 30 --samples-per-epoch 16384

    # History-only operational control.
    python experiments/catchment_foundation/run_crossformer.py \
        --name crossformer_history_v1 --history-only --epochs 30 --samples-per-epoch 16384
"""
import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.catchment_foundation.run_training import (HORIZON_HOURS, MANIFEST_DEFAULT,
                                                            SCALED_COLS, SPINUP_HOURS,
                                                            TEST_START, TRAIN_END, load_env)

# Every learned time-varying input is standardized from train-period statistics in the manifest.
# The raw temperature/shortwave duplicates and ODE-only physical channels are intentionally absent.
DIRECT_RELEVANT_COLS = ["cfs"] + SCALED_COLS


def build_crossformer_params(
        manifest_path: str, run_name: str, epochs: int, batch_size: int,
        samples_per_epoch: int, max_basins: Optional[int], lr: float, use_wandb: bool,
        patience: int = 5, d_model: int = 64, d_ff: int = 128, n_heads: int = 4,
        e_layers: int = 2, seg_len: int = 24, dropout: float = 0.1,
        context_channels: int = 8, use_future_forcing: bool = True,
        input_clip: Optional[float] = 20.0, loss: str = "mse",
        huber_beta: float = 1.0, residual_smoothing_hours: int = 1,
        nonnegative: bool = False, event_sample_power: float = 0.0,
        require_pretrained_embedding: bool = True) -> Dict:
    """
    Builds an FF-native config for the direct multi-basin Crossformer.

    :param manifest_path: Catchment manifest path.
    :type manifest_path: str
    :param run_name: Output directory and W&B run name.
    :type run_name: str
    :param epochs: Maximum training epochs.
    :type epochs: int
    :param batch_size: Windows per optimization batch.
    :type batch_size: int
    :param samples_per_epoch: Weighted-sampler draws per epoch.
    :type samples_per_epoch: int
    :param max_basins: Optional training-basin cap for smoke runs.
    :type max_basins: int, optional
    :param lr: Adam learning rate.
    :type lr: float
    :param use_wandb: Whether to monitor with W&B.
    :type use_wandb: bool
    :param patience: Early-stopping patience, defaults to 5.
    :type patience: int, optional
    :param d_model: Crossformer model width, defaults to 64.
    :type d_model: int, optional
    :param d_ff: Crossformer feed-forward width, defaults to 128.
    :type d_ff: int, optional
    :param n_heads: Attention head count, defaults to 4.
    :type n_heads: int, optional
    :param e_layers: Encoder depth, defaults to 2.
    :type e_layers: int, optional
    :param seg_len: Segment length in hours, defaults to 24.
    :type seg_len: int, optional
    :param dropout: Dropout probability, defaults to 0.1.
    :type dropout: float, optional
    :param context_channels: Projected catchment-context channels; zero disables context.
    :type context_channels: int, optional
    :param use_future_forcing: Whether the model sees realized horizon meteorology.
    :type use_future_forcing: bool, optional
    :param input_clip: Absolute standardized-input clip; None disables it.
    :type input_clip: float, optional
    :param loss: Pointwise training loss, ``"mse"`` or ``"huber"``, defaults to ``"mse"``.
    :type loss: str, optional
    :param huber_beta: Smooth-L1 transition point when ``loss="huber"``, defaults to 1.0.
    :type huber_beta: float, optional
    :param residual_smoothing_hours: Moving-average width on the predicted residual, defaults to 1.
    :type residual_smoothing_hours: int, optional
    :param nonnegative: Whether to constrain predicted flow to be nonnegative, defaults to False.
    :type nonnegative: bool, optional
    :param event_sample_power: Exponent on horizon-variance sampling; zero samples quiet and
        event windows uniformly, defaults to 0.0.
    :type event_sample_power: float, optional
    :param require_pretrained_embedding: Whether to exclude basins without a pretrained catchment
        embedding, defaults to True so fixed contrastive vectors are not mixed with learned IDs.
    :type require_pretrained_embedding: bool, optional
    :return: Resolved Flow Forecast configuration.
    :rtype: Dict
    """
    forcing_tag = "hindcast_forcing" if use_future_forcing else "history_only"
    context_tag = "catchment_context" if context_channels else "no_catchment_context"
    params = {
        "model_name": "CrossformerMultiBasin",
        "model_type": "PyTorch",
        "model_params": {
            "n_time_series": len(DIRECT_RELEVANT_COLS) + 1,
            "spinup_length": SPINUP_HOURS,
            "forecast_length": HORIZON_HOURS,
            "basin_info_path": manifest_path,
            "seg_len": seg_len,
            "win_size": 4,
            "factor": 10,
            "d_model": d_model,
            "d_ff": d_ff,
            "n_heads": n_heads,
            "e_layers": e_layers,
            "dropout": dropout,
            "context_dim": 256,
            "context_channels": context_channels,
            "use_future_forcing": use_future_forcing,
            "input_clip": input_clip,
            "residual_smoothing_hours": residual_smoothing_hours,
            "nonnegative": nonnegative,
        },
        "dataset_params": {
            "class": "MultiBasinCatchmentWindow",
            "training_path": manifest_path,
            "validation_path": manifest_path,
            "test_path": manifest_path,
            "batch_size": batch_size,
            "num_workers": 0,
            "forecast_history": SPINUP_HOURS,
            "forecast_length": HORIZON_HOURS,
            "target_col": ["cfs"],
            "relevant_cols": DIRECT_RELEVANT_COLS,
            "scaled_cols": SCALED_COLS,
            "window_stride": 72,
            "min_valid_fraction": 0.95,
            "train_basin_split": "train",
            "valid_basin_split": "train",
            "test_basin_split": "train",
            "train_end_date": TRAIN_END,
            "valid_start_date": TRAIN_END,
            "valid_end_date": TEST_START,
            "test_start_date": TEST_START,
            "valid_window_stride": 336,
            "test_window_stride": 672,
            "train_samples_per_epoch": samples_per_epoch,
            "event_sample_power": event_sample_power,
            "require_pretrained_embedding": require_pretrained_embedding,
        },
        "early_stopping": {"patience": patience},
        "training_params": {
            "criterion": "SmoothL1Loss" if loss == "huber" else "MSE",
            "optimizer": "Adam",
            "optim_params": {"lr": lr},
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_grad_norm": 1.0,
        },
        "GCS": False,
        "wandb": {
            "project": "catchment-foundation",
            "name": run_name,
            "tags": ["multi_basin", "direct_crossformer", forcing_tag, context_tag],
        } if use_wandb else False,
        "forward_params": {},
        "metrics": ["MSE"],
    }
    if loss == "huber":
        params["training_params"]["criterion_params"] = {"beta": huber_beta}
    if params["wandb"]:
        params["wandb"]["tags"].extend([
            loss,
            "smooth_%dh" % residual_smoothing_hours,
            "nonnegative" if nonnegative else "unconstrained",
            "event_power_%g" % event_sample_power,
            "embedded_only" if require_pretrained_embedding else "mixed_context",
        ])
    if max_basins is not None:
        params["dataset_params"]["max_basins"] = max_basins
    return params


def main() -> None:
    """Builds, trains, evaluates and optionally compares the direct Crossformer control."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="W&B run name and output directory")
    parser.add_argument("--manifest", default=MANIFEST_DEFAULT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=16384)
    parser.add_argument("--max-basins", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seg-len", type=int, default=24,
                        help="Crossformer segment length in hours")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--context-channels", type=int, default=8,
                        help="Projected catchment-embedding channels; 0 disables context")
    parser.add_argument("--history-only", action="store_true",
                        help="Do not expose realized horizon meteorology to the model")
    parser.add_argument("--input-clip", type=float, default=20.0,
                        help="Clip standardized inputs to +/- this value; <=0 disables")
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--residual-smoothing-hours", type=int, default=1,
                        help="Moving-average width on Crossformer residual; 1 disables")
    parser.add_argument("--nonnegative", action="store_true",
                        help="Apply a smooth nonnegative projection to final flow")
    parser.add_argument("--event-sample-power", type=float, default=0.0,
                        help="Horizon-variance sampler exponent; default 0 samples windows "
                             "uniformly; 1 restores the original event emphasis")
    parser.add_argument(
        "--allow-missing-embeddings", action="store_true",
        help="Include basins without pretrained embeddings using learned basin-ID rows; "
             "disabled by default because the representation spaces are inconsistent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None,
                        help="Optional torch CPU thread count")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--eval-stride", type=int, default=336)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--compare-to", default=None,
                        help="Optional prior run directory to compare after evaluation")
    args = parser.parse_args()

    load_env(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    os.environ.setdefault("WANDB_ENTITY", "igodfried")

    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    run_dir = os.path.join(os.path.dirname(__file__), "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    input_clip = args.input_clip if args.input_clip > 0 else None
    params = build_crossformer_params(
        args.manifest, args.name, args.epochs, args.batch_size, args.samples_per_epoch,
        args.max_basins, args.lr, not args.no_wandb, patience=args.patience,
        d_model=args.d_model, d_ff=args.d_ff, n_heads=args.heads, e_layers=args.layers,
        seg_len=args.seg_len, dropout=args.dropout, context_channels=args.context_channels,
        use_future_forcing=not args.history_only, input_clip=input_clip, loss=args.loss,
        huber_beta=args.huber_beta,
        residual_smoothing_hours=args.residual_smoothing_hours,
        nonnegative=args.nonnegative,
        event_sample_power=args.event_sample_power,
        require_pretrained_embedding=not args.allow_missing_embeddings)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(params, f, indent=1)

    from flood_forecast.pytorch_training import train_transformer_style
    from flood_forecast.time_model import PyTorchForecast

    model = PyTorchForecast(params["model_name"], args.manifest, args.manifest, args.manifest,
                            params)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    print("Crossformer mode: %s; catchment context: %s; parameters: %s"
          % ("hindcast (realized future met)" if not args.history_only else "history only",
             "on" if args.context_channels else "off", f"{parameter_count:,}"))
    print("train windows: %d (%d basins), valid windows: %d, test windows: %d"
          % (len(model.training), len(model.training.basin_loaders), len(model.validation),
             len(model.test_data)))
    train_transformer_style(model, params["training_params"], forward_params={},
                            model_filepath=run_dir)
    print("Training complete; model and run-scoped checkpoint saved under %s" % run_dir)

    if not args.skip_eval:
        from experiments.catchment_foundation.evaluate import evaluate_splits

        evaluate_splits(model, args.manifest, run_dir, eval_stride=args.eval_stride,
                        max_basins=args.max_basins)
        if args.compare_to:
            from experiments.catchment_foundation.compare_runs import compare_run_directories

            comparison = compare_run_directories(args.compare_to, run_dir)
            comparison_path = os.path.join(run_dir, "comparison.json")
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2)
            print(json.dumps(comparison, indent=2))
            print("Comparison written to %s" % comparison_path)


if __name__ == "__main__":
    main()
