# Add multi-basin catchment foundation model support

## Summary

This branch adds a hybrid physics/ML hydrology forecaster that trains one shared model
across many Colorado catchments at once, plus the pipeline to build its input manifest,
train it, and evaluate it on gauged and ungauged basins. It also generalizes the existing
multimodal/contrastive meta-model code into reusable components, and fixes an early-stopping
bookkeeping bug and a stray debug file write uncovered along the way.

Relative to `master`, this branch is 6 commits ahead and touches 25 files
(+6,879 / −324 lines).

## New components

- **`experiments/catchment_foundation/`** — the end-to-end pipeline:
  - `build_manifest.py`: scans the Water repo's scraped CO gauge registry and writes a
    manifest JSON per basin (csv path, drainage area, lapse-rate temperature offset,
    train-period flow scale and met normalization stats, embedding availability, optional
    SNODAS SWE path) plus a shared `preprocessing` block, and assigns an ungauged-holdout
    split.
  - `run_training.py`: builds a flow-forecast config for `HybridGR4MultiBasin` +
    `MultiBasinCatchmentWindow` and trains via the existing `train_transformer_style` loop,
    with an optional `--swe` flag to add a SNODAS snow-water-equivalent input channel.
  - `evaluate.py`: post-training evaluation on two splits — `gauged_2023` (trained basins,
    2023+ held-out time) and `ungauged_2023` (basins never trained on) — reporting pooled
    and per-basin skill-vs-persistence metrics.
- **`flood_forecast/ode/physics/forecast_training.py`** (new, 533 lines): `HybridGR4Forecast`
  (single-basin) and `HybridGR4MultiBasin` (manifest-driven, per-basin context + flow scale
  selected via a basin-index channel) wrapping `HybridGR4Model` in the standard FF
  `forward(x)` interface, plus the shared `forecast_report` evaluation/plotting utility.
  Both models are registered in `flood_forecast/model_dict_function.py`
  (`"HybridGR4"`, `"HybridGR4MultiBasin"`), which also newly registers `NSELoss` and
  `MaskedMSELoss` from `flood_forecast/custom/custom_opt.py`.
- **`flood_forecast/preprocessing/pytorch_loaders.py`**: new `MultiBasinWindowLoader`
  (combines per-basin `CatchmentWindowLoader`s, applies manifest preprocessing, standardizes
  only requested columns, appends a basin-index channel, exposes variance/frequency-weighted
  `sample_weights`). `flood_forecast/time_model.py` wires it up as dataset class
  `"MultiBasinCatchmentWindow"` (and adds a plain `"CatchmentWindow"` single-basin class too).
- **`flood_forecast/pytorch_training.py`**: `train_transformer_style` now builds a
  `WeightedRandomSampler` from a dataset's `sample_weights`/`samples_per_epoch` when present
  (used by `MultiBasinWindowLoader`), and `handle_meta_data` gained an `embedding_path`
  config style that loads a precomputed entity embedding instead of training a meta-model.
- **`flood_forecast/meta_models/contrastive_train.py`** and **`multimodal_encoder.py`**
  (new): generalize the multimodal/contrastive pretraining code (previously in
  `flood_forecast/multi_models/`) into reusable `MultiModalEncoder` /
  `StaticEmbeddingMetaModel` components, with `load_embedding`/`save_embeddings` helpers.
  `flood_forecast/multi_models/catchment_embedding.py` and `contrastive_pretrain.py` were
  updated accordingly.
- **`flood_forecast/ode/physics/hydrology.py`**: adds `GR4SnowDynamics` and
  `GR4SnowParameterHead`, now exported from `flood_forecast/ode/physics/__init__.py`.
- **`.circleci/config.yml`**: registers the four new test files (`test_multimodal_encoder.py`,
  `test_hybrid_gr4.py`, `test_forecast_training.py`, `test_multi_basin_forecast.py`) as
  explicit `coverage run` steps.

## Bug fixes

- **Early-stopping best-score bug (`flood_forecast/training_utils.py`)**: `EarlyStopper
  .check_loss` previously updated `self.best_score = score` inside the *non-improvement*
  branch whenever `cumulative_delta` was falsy and `score > self.best_score` — i.e. it could
  raise the "best" bar to a numerically larger (worse) loss than the last checkpointed one,
  even though no new checkpoint was saved for that score. Over time this let the effective
  improvement threshold drift toward worse models, undermining early stopping. The fix
  removes that branch entirely: `best_score` is now only ever updated when a checkpoint is
  actually saved (the `elif` was also flipped to the equivalent `score >= self.best_score -
  min_delta` form for the same non-improvement condition), and the counter's print statement
  was made more descriptive (`"Early stopping counter %d of %d"` instead of a bare number).
  Related: `train_transformer_style` (`flood_forecast/pytorch_training.py`) now also
  restores the best checkpoint when training runs to `max_epochs` without ever tripping the
  patience counter, instead of leaving the possibly-past-peak final-epoch weights loaded.
- **Stray `temp_df.csv` write removed**: `CSVDataLoader.__init__`
  (`flood_forecast/preprocessing/pytorch_loaders.py`) unconditionally wrote the full,
  post-scaling dataframe to `temp_df.csv` in the working directory on every instantiation.
  This debug leftover is removed.

## Test plan

- [ ] `coverage run -m unittest -v tests/test_hybrid_gr4.py`
- [ ] `coverage run -m unittest -v tests/test_forecast_training.py`
- [ ] `coverage run -m unittest -v tests/test_multi_basin_forecast.py`
- [ ] `coverage run -m unittest -v tests/test_multimodal_encoder.py`
- [ ] `coverage run -m unittest -v tests/test_contrastive_pretrain.py`
- [ ] Full CircleCI suite green (new tests are now wired into `.circleci/config.yml`)
- [ ] Manual smoke run of `experiments/catchment_foundation/run_training.py --max-basins 10
      --epochs 3 --samples-per-epoch 512 --no-wandb` against a small manifest to confirm the
      end-to-end pipeline (manifest → train → evaluate) still runs

🤖 Generated with [Claude Code](https://claude.com/claude-code)
