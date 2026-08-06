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
# Physical channels consumed by the ODE, never scaled and never fed to the learned encoder.
PHYS_COLS = ["precip_raw", "pet_raw", "asos_raw", "asos_observed"]
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


def make_basin_validation_manifest(manifest_path: str, output_path: str, count: int,
                                   seed: int,
                                   require_pretrained_embedding: bool = False) -> list:
    """
    Writes a derived manifest with whole training basins reserved for development validation.

    The original ``holdout`` basins are never eligible. This gives early stopping an honest
    catchment-transfer signal instead of validating only on later years from basins whose
    hypernetwork contexts were already fitted.

    :param manifest_path: Source manifest path.
    :type manifest_path: str
    :param output_path: Derived manifest path to write.
    :type output_path: str
    :param count: Number of training basins to re-label ``basin_valid``.
    :type count: int
    :param seed: Deterministic selection seed.
    :type seed: int
    :param require_pretrained_embedding: Restrict candidates to basins with pretrained contexts,
        defaults to False.
    :type require_pretrained_embedding: bool, optional
    :return: Selected validation site IDs.
    :rtype: list
    """
    if count <= 0:
        raise ValueError("count must be positive")
    import numpy as np

    with open(manifest_path) as f:
        manifest = json.load(f)
    candidates = [
        index for index, basin in enumerate(manifest["basins"])
        if basin.get("split") == "train"
        and (not require_pretrained_embedding or basin.get("has_embedding", False))
    ]
    if count >= len(candidates):
        raise ValueError("basin validation count must leave at least one eligible training basin")
    rng = np.random.default_rng(seed)
    selected_indices = set(rng.choice(candidates, size=count, replace=False).tolist())
    selected_sites = []
    for index in selected_indices:
        basin = manifest["basins"][index]
        basin["source_split"] = basin.get("split")
        basin["split"] = "basin_valid"
        selected_sites.append(basin["site_id"])
    manifest["basin_validation"] = {
        "count": count,
        "seed": seed,
        "require_pretrained_embedding": require_pretrained_embedding,
        "site_ids": sorted(selected_sites),
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=1)
    return sorted(selected_sites)


def build_params(manifest_path: str, run_name: str, epochs: int, batch_size: int,
                 samples_per_epoch: int, max_basins: Optional[int], lr: float,
                 use_wandb: bool, use_swe: bool = False, patience: int = 3,
                 anchored: bool = False, use_multiplier: bool = True,
                 use_asos_gate: bool = False, normalize_context: bool = False,
                 parameter_logit_limit: Optional[float] = None,
                 x4_max: Optional[float] = None,
                 event_sample_power: float = 1.0,
                 require_pretrained_embedding: bool = False,
                 seed: int = 42, num_workers: int = 0,
                 production_store_init_fraction: float = 0.6,
                 swe_supervision_weight: float = 0.0,
                 decouple_snow_head: bool = False,
                 valid_basin_split: str = "train") -> Dict:
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
    :param patience: Early-stopping patience in epochs, defaults to 3.
    :type patience: int, optional
    :param anchored: Whether the ODE is driven by physical gridded forcing that the network may
        only correct, rather than by forcing the network invents, defaults to False.
    :type anchored: bool, optional
    :param use_multiplier: Anchored mode: learn the bounded precipitation multiplier (rung 2 of
        the ladder; False gives the pure-physics rung 1), defaults to True.
    :type use_multiplier: bool, optional
    :param use_asos_gate: Anchored mode: add the gated station-innovation term (rung 3),
        defaults to False.
    :type use_asos_gate: bool, optional
    :return: The config dict.
    :rtype: Dict
    """
    relevant_cols = RELEVANT_COLS + (["snodas_swe_mm"] if use_swe else [])
    if anchored:
        # Physical (unscaled) channels for the anchored forcing path, in the order the model
        # expects: [gridded precip, gridded PET, station precip, station-observed mask].
        relevant_cols = relevant_cols + PHYS_COLS
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
            "normalize_context": normalize_context,
            "production_store_init_fraction": production_store_init_fraction,
            "swe_supervision_weight": swe_supervision_weight,
        },
        "dataset_params": {
            "class": "MultiBasinCatchmentWindow",
            "training_path": manifest_path, "validation_path": manifest_path,
            "test_path": manifest_path,
            "batch_size": batch_size,
            "forecast_history": SPINUP_HOURS, "forecast_length": HORIZON_HOURS,
            "target_col": ["cfs"], "relevant_cols": relevant_cols, "scaled_cols": SCALED_COLS,
            "window_stride": 72, "min_valid_fraction": 0.95,
            "train_basin_split": "train", "valid_basin_split": valid_basin_split,
            "test_basin_split": "train",
            "train_end_date": TRAIN_END,
            "valid_start_date": TRAIN_END, "valid_end_date": TEST_START,
            "test_start_date": TEST_START,
            "valid_window_stride": 336, "test_window_stride": 672,
            "train_samples_per_epoch": samples_per_epoch,
            "event_sample_power": event_sample_power,
            "require_pretrained_embedding": require_pretrained_embedding,
            "num_workers": num_workers,
        },
        "early_stopping": {"patience": patience},
        # FF builds the optimizer from optim_params ONLY; a top-level "lr" key is ignored, so the
        # learning rate must go here or every run silently uses the optimizer's default.
        "training_params": {"criterion": "MSE", "optimizer": "Adam",
                            "optim_params": {"lr": lr},
                            "lr": lr, "epochs": epochs, "batch_size": batch_size,
                            "max_grad_norm": 1.0},
        "GCS": False,
        "wandb": {"project": "catchment-foundation", "name": run_name,
                  "tags": ["multi_basin", "hybrid_gr4"]} if use_wandb else False,
        "forward_params": {},
        "metrics": ["MSE"],
        "seed": seed,
    }
    parameter_head_params = {}
    if parameter_logit_limit is not None:
        parameter_head_params["logit_limit"] = parameter_logit_limit
    if x4_max is not None:
        parameter_head_params["x4_range"] = (2.0, x4_max)
    if decouple_snow_head:
        parameter_head_params["decouple_snow_head"] = True
    if parameter_head_params:
        params["model_params"]["parameter_head_params"] = parameter_head_params
    if use_swe:
        params["model_params"]["swe_index"] = relevant_cols.index("snodas_swe_mm")
        if params["wandb"]:
            params["wandb"]["tags"].append("snodas_swe")
    if anchored:
        params["model_params"].update({
            "anchored": True, "use_multiplier": use_multiplier, "use_asos_gate": use_asos_gate,
            "phys_indices": [relevant_cols.index(c) for c in PHYS_COLS]})
        if params["wandb"]:
            rung = "physics" if not use_multiplier else (
                "multiplier_asos" if use_asos_gate else "multiplier")
            params["wandb"]["tags"].extend(["anchored", rung])
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
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stopping patience in epochs")
    parser.add_argument("--anchored", action="store_true",
                        help="Drive the ODE with physical gridded forcing the network may only "
                             "correct, instead of forcing the network invents")
    parser.add_argument("--no-multiplier", action="store_true",
                        help="Anchored mode: disable the learned multiplier (pure-physics rung)")
    parser.add_argument("--asos-gate", action="store_true",
                        help="Anchored mode: add the gated ASOS station-innovation term")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for torch/numpy and the weighted sampler, so that runs "
                             "differing in one config option are otherwise identical")
    parser.add_argument("--normalize-context", action="store_true",
                        help="L2-normalize pretrained and learned catchment contexts")
    parser.add_argument("--parameter-logit-limit", type=float, default=None,
                        help="Smoothly keep GR4/snow parameter logits within +/- this value")
    parser.add_argument("--x4-max", type=float, default=None,
                        help="Override the GR4 routing time upper bound (hours)")
    parser.add_argument("--event-sample-power", type=float, default=1.0,
                        help="Exponent on within-basin event-variance sampling; 0 is uniform")
    parser.add_argument("--require-embeddings", action="store_true",
                        help="Train/evaluate only basins with pretrained catchment embeddings")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes; zero avoids macOS shared-memory crashes")
    parser.add_argument("--production-init-fraction", type=float, default=0.6,
                        help="Initial production-store fraction of X1 before the 30-day spin-up")
    parser.add_argument("--swe-loss-weight", type=float, default=0.0,
                        help="Weight on log-SWE auxiliary supervision during training")
    parser.add_argument("--decouple-snow-head", action="store_true",
                        help="Use a separate catchment projection for snow parameters")
    parser.add_argument("--basin-validation-count", type=int, default=0,
                        help="Reserve this many whole training basins for early stopping")
    parser.add_argument("--manifest-validation", action="store_true",
                        help="Use basin_valid split labels already present in the manifest "
                             "(e.g. carried over by build_manifest --carry-splits) instead of "
                             "sampling a fresh development holdout")
    parser.add_argument("--init-from", default=None,
                        help="Optional model state_dict used to warm-start this run")
    parser.add_argument("--freeze-parameter-head", action="store_true",
                        help="Freeze the GR4/snow parameter head after optional warm start")
    parser.add_argument("--init-parameter-head", default=None,
                        help="GR4SnowParameterHead state_dict from pretrain_parameter_head.py")
    parser.add_argument("--init-contexts", default=None,
                        help="Inverted context vectors ({site_ids, contexts}) used to warm-start "
                             "learnable context rows of basins without pretrained embeddings")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--eval-stride", type=int, default=336,
                        help="Window stride (hours) for the post-training forecast_report eval")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-gauged-only", action="store_true",
                        help="During iteration, skip the ungauged post-training evaluation")
    parser.add_argument("--eval-basin-validation-only", action="store_true",
                        help="Evaluate only the development basin holdout; do not inspect final test")
    args = parser.parse_args()
    if args.eval_gauged_only and args.eval_basin_validation_only:
        parser.error("choose at most one restricted evaluation split")
    if args.eval_basin_validation_only and args.basin_validation_count <= 0 \
            and not args.manifest_validation:
        parser.error("--eval-basin-validation-only requires --basin-validation-count or "
                     "--manifest-validation")
    if args.manifest_validation and args.basin_validation_count > 0:
        parser.error("choose either --manifest-validation or --basin-validation-count")

    load_env(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    os.environ.setdefault("WANDB_ENTITY", "igodfried")

    import numpy as np
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = os.path.join(os.path.dirname(__file__), "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    training_manifest = args.manifest
    if args.basin_validation_count > 0:
        training_manifest = os.path.join(run_dir, "development_manifest.json")
        selected = make_basin_validation_manifest(
            args.manifest, training_manifest, args.basin_validation_count, args.seed,
            require_pretrained_embedding=args.require_embeddings)
        print("development basin validation: %s" % ", ".join(selected))
    params = build_params(training_manifest, args.name, args.epochs, args.batch_size,
                          args.samples_per_epoch, args.max_basins, args.lr, not args.no_wandb,
                          use_swe=args.swe, patience=args.patience, anchored=args.anchored,
                          use_multiplier=not args.no_multiplier, use_asos_gate=args.asos_gate,
                          normalize_context=args.normalize_context,
                          parameter_logit_limit=args.parameter_logit_limit,
                          x4_max=args.x4_max,
                          event_sample_power=args.event_sample_power,
                          require_pretrained_embedding=args.require_embeddings,
                          seed=args.seed, num_workers=args.num_workers,
                          production_store_init_fraction=args.production_init_fraction,
                          swe_supervision_weight=args.swe_loss_weight,
                          decouple_snow_head=args.decouple_snow_head,
                          valid_basin_split=("basin_valid"
                                             if args.basin_validation_count > 0
                                             or args.manifest_validation else "train"))
    params["training_params"]["freeze_parameter_head"] = args.freeze_parameter_head
    params["training_params"]["init_from"] = args.init_from
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(params, f, indent=1)

    from flood_forecast.pytorch_training import train_transformer_style
    from flood_forecast.time_model import PyTorchForecast

    model = PyTorchForecast(params["model_name"], training_manifest, training_manifest,
                            training_manifest, params)
    if args.init_from:
        state = torch.load(args.init_from, map_location=model.device, weights_only=True)
        model.model.load_state_dict(state)
    if args.init_parameter_head:
        head_state = torch.load(args.init_parameter_head, map_location=model.device,
                                weights_only=True)
        model.model.hybrid.parameter_head.load_state_dict(head_state)
        print("parameter head warm-started from %s" % args.init_parameter_head)
    if args.init_contexts:
        payload = torch.load(args.init_contexts, map_location=model.device, weights_only=True)
        with open(training_manifest) as f:
            positions = {b["site_id"]: i for i, b in enumerate(json.load(f)["basins"])}
        initialized = 0
        with torch.no_grad():
            for site, vector in zip(payload["site_ids"], payload["contexts"]):
                position = positions.get(site)
                if position is not None and not bool(model.model.has_fixed_context[position]):
                    model.model.learned_context.weight[position] = vector
                    initialized += 1
        print("warm-started %d learnable context rows from %s"
              % (initialized, args.init_contexts))
    if args.freeze_parameter_head:
        for parameter in model.model.hybrid.parameter_head.parameters():
            parameter.requires_grad_(False)
    print("train windows: %d (%d basins), valid windows: %d, test windows: %d"
          % (len(model.training), len(model.training.basin_loaders), len(model.validation),
             len(model.test_data)))
    train_transformer_style(model, params["training_params"], forward_params={},
                            model_filepath=run_dir)
    print("Training complete; model saved under %s" % run_dir)

    from hybrid_diagnostics import save_parameter_diagnostics
    save_parameter_diagnostics(model.model, training_manifest, run_dir,
                               model.training.basin_positions)

    if not args.skip_eval:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from evaluate import evaluate_splits
        split_names = None
        if args.eval_gauged_only:
            split_names = ["gauged_2023"]
        elif args.eval_basin_validation_only:
            split_names = ["basin_valid_2023"]
        evaluate_splits(model, training_manifest, run_dir, eval_stride=args.eval_stride,
                        max_basins=args.max_basins,
                        split_names=split_names)


if __name__ == "__main__":
    main()
