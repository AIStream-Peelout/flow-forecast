# Catchment Foundation Model — Multi-Basin Training Pipeline

This directory holds the scripts for training and evaluating a single hybrid GR4/ODE
hydrology model across many Colorado catchments at once (`HybridGR4MultiBasin`), instead
of the one-model-per-basin `HybridGR4Forecast` setup. The pipeline has three stages:

1. **`build_manifest.py`** — scans the Water repo's scraped gauge data and writes a JSON
   manifest describing every usable basin (paths, area, normalization stats, splits).
2. **`run_training.py`** — builds a flow-forecast config around that manifest and trains
   `HybridGR4MultiBasin` through the normal `train_transformer_style` loop.
3. **`evaluate.py`** — runs the trained model over two held-out splits and produces
   per-basin and pooled skill metrics.

## Data prerequisites

The manifest builder reads directly from the **Water** repo's pilot data (not this repo):

- `~/Documents/GitHub/Water/pilot_data/scrapes/CO/<site_id>/<site_id>_static.json` — static
  basin attributes (drainage area, mean basin elevation, gauge altitude).
- `~/Documents/GitHub/Water/pilot_data/scrapes/CO/<site_id>/<site_id>_hourly_full.csv` — the
  hourly gauge/met bundle (`datetime`, `cfs`, `precipitation`, `temperature`,
  `shortwave_radiation`, `longwave_radiation`, `specific_humidity`, `wind_east`,
  `wind_north`, optional `p01m`, `pet_mm_hr`).
- `~/Documents/GitHub/Water/pilot_data/scrapes/CO/registry.json` — scrape status registry;
  only sites marked `"status": "completed"` are considered.
- `~/Documents/GitHub/Water/pilot_data/embedding_dataset/CO/embeddings_concat.pt` — the
  pretrained per-basin embedding bank (`{"site_ids": [...], "embeddings": Tensor}`).
- `~/Documents/GitHub/Water/pilot_data/snodas_series/CO/<site_id>_snodas_swe.csv` — optional
  daily basin-mean SNODAS snow-water-equivalent series, built by
  `Water/snodas_series_scrape.py`. Only referenced from the manifest when present for a site.

## Stage 1: build the manifest

```bash
python experiments/catchment_foundation/build_manifest.py \
    --out experiments/catchment_foundation/manifests/co_manifest.json
```

Optional flags: `--n-holdout` (basins fully held out for ungauged-generalization testing,
default 15), `--min-train-years` (minimum valid pre-cutoff flow years to keep a gauge,
default 2.0), `--seed` (holdout-selection RNG seed, default 42).

For each completed gauge, `build_basin_entry` requires the static drainage area (falling
back to `drain_area_va` in square miles when GAGES-II area is missing) and all the required
met columns (everything in `MET_COLS` except `p01m`, which can be backfilled from
`precipitation`). It then:

- computes a lapse-rate temperature offset `temp_offset_c = -6.5 * (basin_mean_elev_m -
  gauge_elev_m) / 1000` from GAGES-II basin elevation vs. gauge altitude;
- computes `flow_scale_mm_hr`, the standard deviation of pre-`TRAIN_END` (2022-01-01) flow
  converted from cfs to mm/hr via the basin area, used later for per-basin flow
  standardization;
- computes per-column `met_stats` (mean, std) over the same pre-cutoff training period;
- skips the basin entirely if drainage area, flow variance, or any met column's stats are
  missing or degenerate.

Gauges that pass all the above and have both a pretrained embedding and enough 2023+ data
(`> 24 * 120` hourly rows) are eligible for the **ungauged holdout**: `--n-holdout` of them
are sampled (without replacement) and marked `"split": "holdout"`; every other basin is
`"split": "train"`.

### Manifest JSON schema

```json
{
  "embedding_path": "~/Documents/GitHub/Water/pilot_data/embedding_dataset/CO/embeddings_concat.pt",
  "train_end": "2022-01-01",
  "preprocessing": {
    "fill_from": {"p01m": "precipitation"},
    "copy_cols": {"sw_raw": "shortwave_radiation"},
    "lapse": {"source": "temperature", "target": "temp_lapse_k"},
    "swe_col": "snodas_swe_mm"
  },
  "basins": [
    {
      "site_id": "06730500",
      "csv_path": ".../scrapes/CO/06730500/06730500_hourly_full.csv",
      "area_sq_km": 1234.5,
      "temp_offset_c": -1.2345,
      "flow_scale_mm_hr": 0.0873,
      "met_stats": {
        "precipitation": [0.012, 0.045],
        "temperature": [280.3, 9.8],
        "shortwave_radiation": [180.2, 210.5],
        "longwave_radiation": [280.1, 40.2],
        "specific_humidity": [0.004, 0.002],
        "wind_east": [0.5, 2.1],
        "wind_north": [0.3, 1.9],
        "p01m": [0.011, 0.04],
        "pet_mm_hr": [0.02, 0.015]
      },
      "has_embedding": true,
      "train_valid_hours": 26280,
      "rows_2023_plus": 8760,
      "split": "train",
      "swe_csv_path": ".../snodas_series/CO/06730500_snodas_swe.csv"
    }
  ]
}
```

Top-level fields:

- `embedding_path` — the pretrained embedding bank, used by `HybridGR4MultiBasin` as a
  fixed catchment context for basins with `has_embedding: true`.
- `train_end` — the pre/post-cutoff boundary used both for manifest stats and for the
  training/validation date split in `run_training.py`.
- `preprocessing` — instructions consumed by `MultiBasinWindowLoader` when it assembles
  each basin's frame:
  - `fill_from` — for each `{derived_col: source_col}`, fill NaNs in `derived_col` from
    `source_col` (creating the column if absent). Here `p01m` (NLDAS hourly precip) is
    backfilled from `precipitation`.
  - `copy_cols` — for each `{new_col: source_col}`, copy a column under a new name. Here
    `sw_raw` duplicates `shortwave_radiation` so an unscaled physical copy survives even
    when `shortwave_radiation` itself is in `scaled_cols`.
  - `lapse` — `target = source + basin["temp_offset_c"]`; here `temp_lapse_k` is the raw
    Kelvin temperature corrected for the basin's lapse-rate offset.
  - `swe_col` — the column name the loader fills with the basin's hourly-resampled SWE
    series (see below), only if `swe_col` also appears in `relevant_cols`.
- `basins` — the per-basin entries described above, keyed by `site_id`.

Per-basin field notes: `csv_path` and (optional) `swe_csv_path` are absolute paths into the
Water repo; `split` is `"train"` or `"holdout"`; `has_embedding` gates whether
`HybridGR4MultiBasin` uses the pretrained embedding or a learnable fallback for that basin.

## Stage 2: train

`run_training.py` builds a standard flow-forecast config dict for `HybridGR4MultiBasin` +
the `MultiBasinCatchmentWindow` dataset class, then hands it to
`flood_forecast.pytorch_training.train_transformer_style` via `PyTorchForecast`.

Fixed constants: `SPINUP_HOURS = 720` (30-day spin-up), `HORIZON_HOURS = 336` (14-day
forecast), `TRAIN_END = "2022-01-01"`, `TEST_START = "2023-01-01"`. Training uses
`window_stride=72`; validation (2022 data) uses `valid_window_stride=336`; the in-loop test
split uses `test_window_stride=672`. `RELEVANT_COLS` is `["cfs", "precipitation",
"temperature", "shortwave_radiation", "longwave_radiation", "specific_humidity",
"wind_east", "wind_north", "p01m", "pet_mm_hr", "temp_lapse_k", "sw_raw"]`, with
`SCALED_COLS` covering only the 9 met columns from `precipitation` through `pet_mm_hr`
(indices 1–9) — `cfs`, `temp_lapse_k`, and `sw_raw` are deliberately left unscaled (see
Gotchas).

Early stopping uses `patience=3`. W&B logging (when enabled) goes to project
`catchment-foundation`, entity `igodfried` (set via `WANDB_ENTITY` env var if not already
set), with run name `--name` and tags `["multi_basin", "hybrid_gr4"]` (plus `"snodas_swe"`
when `--swe` is passed). A `.env` file at the repo root is loaded (without overriding
already-set variables) before training starts, so `WANDB_API_KEY` etc. can live there.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--name` | required | Run name; used for the W&B run name and the output dir under `experiments/catchment_foundation/runs/<name>/`. |
| `--manifest` | `experiments/catchment_foundation/manifests/co_manifest.json` | Manifest JSON path. |
| `--epochs` | 20 | Training epochs. |
| `--batch-size` | 8 | Windows per batch. |
| `--samples-per-epoch` | 4096 | Weighted-sampler draws per epoch. |
| `--max-basins` | None | Caps the number of basins loaded (smoke runs). |
| `--lr` | 3e-3 | Adam learning rate. |
| `--swe` | off | Adds a SNODAS SWE input channel (`snodas_swe_mm`) and seeds the model's snow store from it at spin-up start. |
| `--no-wandb` | off | Disables W&B logging. |
| `--eval-stride` | 336 | Window stride (hours) for the post-training `evaluate.py` pass. |
| `--skip-eval` | off | Skips the automatic post-training evaluation. |

### Quickstart

Smoke run (10 basins, 3 epochs, small sample budget — sanity-checks the pipeline quickly):

```bash
python experiments/catchment_foundation/run_training.py \
    --name smoke10 --max-basins 10 --epochs 3 --samples-per-epoch 512 --no-wandb
```

Smoke run with the SNODAS SWE channel enabled:

```bash
python experiments/catchment_foundation/run_training.py \
    --name smoke10_swe --max-basins 10 --epochs 3 --samples-per-epoch 512 --swe --no-wandb
```

Full fleet run (all train-split basins, W&B logging on):

```bash
python experiments/catchment_foundation/run_training.py \
    --name fleet_v1 --epochs 20 --samples-per-epoch 4096
```

Full fleet run with SWE:

```bash
python experiments/catchment_foundation/run_training.py \
    --name fleet_v1_swe --epochs 20 --samples-per-epoch 4096 --swe
```

Each run writes its resolved config to `experiments/catchment_foundation/runs/<name>/config.json`
and prints the train/valid/test window and basin counts before training starts.

## Direct Crossformer control

`run_crossformer.py` is the non-physics control. It uses the same manifest, multi-basin
loader, train/validation/test dates, flow standardization, FF training loop, W&B project,
evaluator and persistence benchmark as `run_training.py`.
`CrossformerMultiBasin` directly predicts standardized flow as a residual around the last
observed flow; it does not use GR4, snow dynamics, state assimilation or an ODE.

The default is deliberately labelled a **hindcast**: the model sees the same realized
meteorology over the 336-hour horizon that the hybrid currently sees. This holds the
information set constant, but it is not an operational 14-day forecast until those
covariates come from an NWP forecast product. `--history-only` removes all horizon
meteorology and is the operational information control. Do not compare a history-only
score with a hindcast score as though only the model changed.

Small end-to-end smoke run:

```bash
python experiments/catchment_foundation/run_crossformer.py \
    --name crossformer_smoke --max-basins 3 --epochs 3 \
    --samples-per-epoch 128 --no-wandb
```

Fleet-scale hindcast control with W&B monitoring:

```bash
python experiments/catchment_foundation/run_crossformer.py \
    --name crossformer_hindcast_v1 --epochs 30 --samples-per-epoch 16384
```

History-only control:

```bash
python experiments/catchment_foundation/run_crossformer.py \
    --name crossformer_history_v1 --history-only \
    --epochs 30 --samples-per-epoch 16384
```

The direct model uses only the ten learned channels (`cfs` plus the nine standardized met
columns); raw ODE-only duplicates are excluded. Historical flow is divided by the same
per-basin train-period scale as the target. Eight projected catchment-embedding channels
are used by default (`--context-channels 0` disables them), and standardized neural inputs
are clipped at +/-20 by default (`--input-clip 0` disables clipping). The clip and removal
of raw ODE-only channels are model-appropriate preprocessing, so the default comparison
tests the overall modelling approach rather than being a one-variable architecture
ablation.

Direct Crossformer runs sample valid windows uniformly by default
(`--event-sample-power 0`). In the controlled 10-basin tuning ladder, the original
horizon-variance weighting (`--event-sample-power 1`) over-emphasized volatile windows,
produced conspicuous dry-basin blow-ups, and reduced both gauged and holdout skill.
Fractional values restore a milder event emphasis. This default is scoped to
`run_crossformer.py`; the generic multi-basin loader and hybrid experiment retain their
original weighting.

Direct runs also require a pretrained catchment embedding by default. The current manifest
contains 50 embedded training basins and 49 training basins without embeddings. Mixing
fixed contrastively pretrained vectors with arbitrary learned basin-ID rows gives the
shared context projection two incompatible representation spaces, so the latter basins
are excluded until their embeddings are filled in. `--allow-missing-embeddings` restores
the mixed behavior for an explicit control experiment. This filter does not alter manifest
positions and therefore remains compatible with the model's embedding and flow-scale
buffers.

The default context mode deliberately matches the hybrid, but it also inherits the current
embedding-bank audit caveat: those embeddings were built with each site's discharge
history extending into the evaluation era. Therefore the present `ungauged_2023` result is
useful for same-pipeline comparison but is not an honest ungauged-generalization claim.
Use `--context-channels 0` for a context-free control, or rebuild the embedding bank using
only pre-split information before making that claim.

After both models have been evaluated on the same stride, compare their saved artifacts:

```bash
python experiments/catchment_foundation/compare_runs.py \
    experiments/catchment_foundation/runs/HYBRID_RUN \
    experiments/catchment_foundation/runs/CROSSFORMER_RUN
```

The comparison reports pooled-skill delta, median-basin delta and basin win rate. It also
checks persistence MSE equality; if that check fails, the two models were not evaluated on
the same observation windows and the model comparison is not controlled.
`run_crossformer.py --compare-to <HYBRID_RUN>` performs this comparison automatically
after evaluation.

For a hydrograph-level comparison between two evaluated direct runs:

```bash
python experiments/catchment_foundation/plot_forecast_comparison.py \
    experiments/catchment_foundation/runs/REFERENCE_RUN \
    experiments/catchment_foundation/runs/CANDIDATE_RUN \
    --reference-label Reference --candidate-label Candidate
```

The controlled tuning results and plot-based failure diagnosis are recorded in
`CROSSFORMER_TUNING.md`.

## Stage 3: evaluate

By default, `run_training.py` calls `evaluate.evaluate_splits` right after training (skip
with `--skip-eval`, or run `evaluate.py`'s `evaluate_splits` directly against a saved
model). It builds a fresh `MultiBasinWindowLoader` for each of two splits, both starting
2023-01-01:

- **`gauged_2023`** — basins the model *was* trained on (manifest `split == "train"`),
  evaluated on their 2023+ held-out time range.
- **`ungauged_2023`** — the 15 basins the model *never saw during training* (manifest
  `split == "holdout"`), also on 2023+.

For every window in a split, `collect_split_outputs` runs the model, converts standardized
output back to physical mm/hr using each basin's `flow_scales` entry, and also computes a
naive persistence forecast (the last observed flow at issue time, held constant across the
horizon).

### Metrics

Both `pooled_metrics` (pooled across all basins in the split) and
`flood_forecast.ode.physics.forecast_training.forecast_report` (per basin, plus example
forecast plots) compute the same four lead-time bands in hours: `day1-3` (0–72),
`day4-7` (72–168), `day8-14` (168–336), `all` (0–336). For each band:

- `mse_mm_hr2` / `mse_persistence_mm_hr2` — model and persistence MSE in (mm/hr)².
- `skill_vs_persistence_pct` = `100 * (1 - mse_model / mse_persistence)` — the **headline
  metric**. 0% means the model is no better than persistence; positive means real skill.
- `median_window_nse` — median per-window Nash–Sutcliffe efficiency.
- `rmse_cfs` (per-basin only, via `forecast_report`) — RMSE converted to cfs using the
  basin's area.

Per-basin `forecast_report` output (metrics JSON + HTML forecast plots for the largest
observed event and a seasonal spread of examples) is written under
`experiments/catchment_foundation/runs/<name>/eval_<split>/<site_id>/` for the 3 basins
with the largest observed-flow variance in each split; `pooled_metrics.json` and
`per_basin_metrics.json` are written for every basin regardless. When W&B is active, pooled
metrics, a summary block (median basin skill, % of basins with positive skill, basin count),
and a per-basin metrics table are all logged per split.

Every split also writes `eval_<split>/forecast_gallery.png` and
`forecast_gallery_cases.json`. The gallery deliberately includes high-flow cases, the
best/worst persistence-relative forecasts, amplitude blow-ups, and peak-timing failures;
each panel shows observed history, observed future, model output, and persistence. In W&B,
look under the top-level Media keys `hydrograph_gallery_gauged_2023` and
`hydrograph_gallery_ungauged_2023`. The same images are also logged under
`gauged_2023/forecast_gallery` and `ungauged_2023/forecast_gallery`. Individual
interactive examples use keys of the form
`<split>/<site_id>/forecast_<number>_<issue-date>`.

## Data flow: loaders and model

### `MultiBasinWindowLoader` (`flood_forecast/preprocessing/pytorch_loaders.py`)

Combines one `CatchmentWindowLoader` per basin (selected from the manifest, optionally
filtered by `basin_split` and capped by `max_basins`) into a single `Dataset`. For each
basin it builds the feature frame per the manifest's `preprocessing` block (`fill_from`,
`copy_cols`, `lapse`, `swe_col`), slices it to `[start_date, end_date)`, standardizes only
the requested `scaled_cols` with that basin's `met_stats`, and wraps the result in a
`CatchmentWindowLoader` (spin-up length `forecast_history`, horizon `forecast_length`,
`window_stride` between window starts, `min_valid_fraction` gating which windows are
indexed). Remaining short gaps are interpolated so served tensors are always finite.

Each `__getitem__` call appends a constant **basin-index channel** (the basin's position in
the *original, unfiltered* manifest list) to both the source and target tensors, and divides
the target flow column by that basin's `flow_scale_mm_hr` (per-basin flow standardization —
the source/spin-up flow stays in physical mm/hr for the physics model's assimilation step).
The loader also exposes `sample_weights` (horizon-flow-variance-weighted within a basin,
basin-frequency-corrected across basins via `basin_sample_power`) and `samples_per_epoch`,
which `train_transformer_style` uses to build a `WeightedRandomSampler`.

### `HybridGR4Forecast` / `HybridGR4MultiBasin` (`flood_forecast/ode/physics/forecast_training.py`)

`HybridGR4Forecast` wraps the physics-based `HybridGR4Model` to match the standard FF
`forward(x)` interface for a **single** basin: given a window of shape `(batch, spinup +
horizon, n_features)`, it runs a no-grad spin-up over the first `spinup_length` steps,
assimilates the observed flow at issue time, then differentiably simulates the horizon.
Catchment context is either a learnable embedding parameter (default) or a fixed vector
loaded from `context_path`.

`HybridGR4MultiBasin` subclasses it for the manifest-driven multi-basin setting. It reads
the trailing basin-index channel `MultiBasinWindowLoader` appends, and uses it to select, per
sample:

- **Catchment context** — the pretrained embedding (`fixed_context`, loaded from the
  manifest's `embedding_path`) for basins with `has_embedding: true`, falling back to a
  **learnable** `nn.Embedding` row (`learned_context`) otherwise, chosen via `torch.where`
  against a boolean `has_fixed_context` mask.
- **Flow scale** — the basin's `flow_scale_mm_hr` (`flow_scales` buffer), by which the
  simulated flow is divided so the criterion compares standardized values against the
  loader's standardized targets (multiply back by `flow_scales[basin]` to recover mm/hr).

**SNODAS SWE seeding** (`--swe`): when `swe_index` is set, the model reads the SWE channel's
value at the very first spin-up timestep (`x[:, 0, swe_index]`) as `initial_snow`, seeding
the physics model's snow store. `MultiBasinWindowLoader`'s `_swe_column` builds this channel
by forward-filling each basin's daily SNODAS series across its calendar day and marking
hours with no observation (including basins with no SWE series at all) with a **`-1.0`
sentinel**; the model treats the sentinel as "no observation" and falls back to its default
empty-snow-store initialization. The SWE channel is excluded from the learned met forcing
so future SWE values can never leak into the horizon simulation.

## Gotchas

- **Physical channels must stay out of `scaled_cols`.** The physics core needs Kelvin
  temperature and physical shortwave/flow units, not standardized values. In
  `run_training.py`, `SCALED_COLS = RELEVANT_COLS[1:10]` deliberately excludes `cfs` (index
  0, the target) and `temp_lapse_k` / `sw_raw` (indices 10–11, the raw physics inputs
  pointed to by `raw_temp_index` / `raw_sw_index`). If you add columns to `relevant_cols`,
  double-check `scaled_cols` doesn't accidentally capture a physical channel.
- **A `"sweep"` key in the config force-enables W&B.** `PyTorchForecast.wandb_init` (in
  `flood_forecast/time_model.py`) checks `if self.params["wandb"]:` first, but falls
  through to `elif "sweep" in self.params:` — merely having a `"sweep"` key present (any
  value) is enough to initialize W&B, even if `"wandb"` is `False`. This is meant for W&B
  sweep agents (which pre-populate `wandb.config`), but it's easy to trip accidentally.
- **Early-stopping checkpoints are run-scoped.** `train_transformer_style` passes
  `<run_dir>/checkpoint.pth` to `EarlyStopper`, so simultaneous hybrid and Crossformer runs
  cannot overwrite or restore one another's best weights. Direct `EarlyStopper` callers
  still default to the legacy working-directory `checkpoint.pth` for compatibility.
- **New test files are not auto-discovered by CircleCI.** `.circleci/config.yml` enumerates
  each `tests/test_*.py` file explicitly with its own `coverage run -m unittest -v
  tests/test_x.py` line; a new test file (e.g. a future addition alongside
  `tests/test_multi_basin_forecast.py` or `tests/test_forecast_training.py`) will silently
  not run in CI until you add that line yourself.
