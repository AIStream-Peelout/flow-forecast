# Direct Crossformer tuning — 10-basin controlled ladder

This ladder held the basin subset, seed, evaluation windows, persistence baseline, and
basic model size fixed. It was intentionally judged using both pooled metrics and saved
hydrographs. Scores below are pooled skill versus persistence; positive is better.

| Run | Controlled change | Gauged day 1–3 | Gauged all | Holdout day 1–3 | Holdout all |
|---|---|---:|---:|---:|---:|
| [ladder4 base](https://wandb.ai/igodfried/catchment-foundation/runs/bkmf4rho) | Event power 1.0 | -52.7% | -104.5% | -17.4% | -109.4% |
| [ladder5 seg48](https://wandb.ai/igodfried/catchment-foundation/runs/oo810lwo) | Segment 24 → 48 h | -153.5% | -309.6% | -41.2% | -283.8% |
| [ladder6 smooth24](https://wandb.ai/igodfried/catchment-foundation/runs/0lfumuut) | 24 h residual smoothing | -222.6% | -279.7% | -68.0% | -291.3% |
| [ladder7 uniform](https://wandb.ai/igodfried/catchment-foundation/runs/2j0glbvp) | Event power 1.0 → 0.0 | **+2.7%** | **+41.0%** | **-0.3%** | **+2.3%** |
| [ladder8 nonnegative](https://wandb.ai/igodfried/catchment-foundation/runs/7t8ey1k8) | Uniform + nonnegative projection | +2.8% | +23.4% | -1.4% | -1.9% |
| [ladder9 event025](https://wandb.ai/igodfried/catchment-foundation/runs/mzs7w8mj) | Event power 0.25 | -11.3% | +36.8% | -5.9% | -3.0% |
| [ladder10 lr3e4](https://wandb.ai/igodfried/catchment-foundation/runs/4epwlayw) | Uniform + LR 3e-4 | **+9.5%** | +32.8% | -1.3% | -2.4% |

`ladder7_crossformer_uniform` is the best balanced configuration and is the basis for the
new direct-run default. Relative to the base it gained 55.4 percentage points on gauged
day 1–3, 145.5 points on gauged all-horizon skill, 17.1 points on holdout day 1–3, and
111.7 points on holdout all-horizon skill. `ladder10` is preferable only if gauged
day 1–3 is the sole target; it gives back longer-lead and holdout performance.

## What the hydrographs showed

- The base model repeatedly generated roughly 24-hour oscillations with the wrong phase
  and amplitude. On nearly dry basins these became large errors relative to persistence.
- Extending the segment to 48 hours did not remove that pattern.
- Smoothing removed the visible hourly wiggle but replaced it with large, smooth
  dry-basin drift.
- Uniform sampling produced realistic recessions on active basins and eliminated most of
  the catastrophic dry-window behavior. It still misses flash/diurnal peaks and sometimes
  oscillates a few cfs above and below zero on truly dry basins.
- The nonnegative projection removed negative flow but degraded active recession shape
  and holdout performance, so it is available as an option rather than enabled by default.
- The lower learning rate improved gauged short-lead forecasts but degraded the more
  balanced all-lead and holdout result.

## Finding the plots

In W&B, open the run's Media section and filter for the exact top-level keys:

- `hydrograph_gallery_gauged_2023`
- `hydrograph_gallery_ungauged_2023`

The `ladder7` run was backfilled with both keys. New evaluations log them automatically.
They are also saved locally as:

```text
runs/<run>/eval_gauged_2023/forecast_gallery.png
runs/<run>/eval_ungauged_2023/forecast_gallery.png
```

The adjacent `forecast_gallery_cases.json` contains metrics for every evaluated window
and identifies the cases shown. Detailed interactive Plotly forecasts live under each
selected basin directory and include actual history, actual future, the model forecast,
and persistence.

## Interpretation limits

These are controlled 10-basin screening runs, not final fleet-scale estimates. They are
hindcasts using realized horizon meteorology. The current pretrained catchment embedding
bank also includes discharge history extending into the evaluation era, so the
`ungauged_2023` numbers are useful for same-pipeline comparison but are not a clean
ungauged-generalization claim.
