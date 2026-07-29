# Catchment Foundation Model — Status, Experiment Log, and Open Issues

**Prepared for external audit. Last updated 2026-07-28.**

This document is deliberately weighted toward failures, negative results, and unresolved
risks. The headline metrics look good in places; the caveats are the point.

---

## 1. What the system is supposed to do

**Operational task:** hourly streamflow forecasting, 14 days ahead (336-hour horizon), given a
30-day (720-hour) spin-up of observed meteorology and flow, with state assimilation at forecast
issue time. This is *not* seasonal simulation — earlier year-long runs were diagnostics only.

**Target scale:** a single shared model across thousands of USGS gauges (currently 99 training
basins in Colorado; design target 9,000+). Nothing basin-specific may be baked in.

**Two repositories:**

| Repo | Role |
|---|---|
| `Water` | Data acquisition only (USGS, ASOS, SNOTEL/SCAN, NLDAS-2, Sentinel-2, GAGES-II, SNODAS) |
| `flow-forecast` (FF) | Reusable model/training components; hydrology-specific code confined to `flood_forecast/ode/physics/` |

**Model architecture (three modules):**

1. **Catchment embedding** — Sentinel-2 patch ViT + tabular MLP over static attributes +
   flow-history encoder, fused contrastively (CLIP-style). Produces a 256-d per-basin vector.
2. **Forcing generator** — a Crossformer encoder mapping raw meteorology + catchment context to
   "effective" precipitation/evapotranspiration.
3. **Physics core** — rigid state-space GR4 rainfall-runoff ODE (Santos et al. 2018) integrated
   with `torchdiffeq`, plus an EXP-HYDRO snow bucket (Höge et al. 2022, HydroNODE). A
   hypernetwork head emits GR4 parameters from the embedding.

Forecast-time state initialization: routing states from observed flow via bisection assimilation
(`match_current_flow`), snow store seeded from gridded SWE (SNODAS), production store from
spin-up.

---

## 2. Honest current status

**The multi-basin pipeline runs end to end and reports positive pooled skill against
persistence. There is no evidence yet that the learned components have fitted anything.**

The most recent completed full-fleet run (`fleet_v1_swe`) posted +46.7% day 1–3 pooled skill on
gauged basins — but its early-stopping restored the **epoch-0 checkpoint**, i.e. an essentially
untrained network. That result therefore measures *physics + flow assimilation + snow
initialization*, not learning. Visual inspection of predicted-vs-actual plots confirms the
predictions are smooth drift that does not track hydrograph shape.

Every subsequent attempt to train it properly crashed on non-finite gradients until 2026-07-28,
when the cause was identified (unscaled inputs to the Crossformer — see §5.7). The first run
with that fix (`fleet_v5_scaled`) is in progress and **its result is not yet known**.

---

## 3. Data inventory

| Asset | State |
|---|---|
| CO gauge scrapes | 123 completed / 4 failed of 127 registry entries |
| Manifest basins (usable) | 114 → 99 train / 15 ungauged holdout |
| Pretrained embeddings | 65 of 114 basins (rest use learnable embedding rows) |
| SNODAS basin-mean SWE | 117 basin series, 6,630 daily grids, 2003-10-01 → 2026-07-15 |
| Training windows (fleet) | 212,077 across 99 basins (stride 72 h) |
| Eval windows | 6,281 gauged / 1,114 ungauged (2023+) |

SNODAS is fetched only for the Oct 1 – Jul 15 snow season; outside it, CO basin SWE is zero and
the loader's "no observation" sentinel produces the same empty-snow-store initialization.
One date (2003-10-30) has no SWE product upstream and is marked missing.

**Splits:** 15 basins held out entirely (ungauged-generalization test) + 2023-onward held out in
time for the remaining basins. Training data ends 2022-01-01; 2022 is validation.

---

## 4. Experiment log

### 4.1 Embedding pretraining — SUCCEEDED (with a caveat)

154 CO sites, contrastive InfoNCE. Loss 4.4 → 3.05 (chance = ln(64) ≈ 4.16). Clusters recover
hydrologic regimes (plains/high-drainage vs. alpine snowmelt headwaters vs. mainstems).
Vision-only projection PC1 correlates 0.65 with basin mean elevation.

*Caveat:* the full embedding includes static attributes that contain elevation and snow
fraction, so only the vision-only correlation is genuinely novel signal.

### 4.2 Single-basin hybrid model (Cache la Poudre, 06752260) — FAILED, arc closed

| Attempt | Result | Diagnosis |
|---|---|---|
| Synthetic overfit gate | NSE ≈ 0.8 | Passes only with fast `x4` range (0.5–24 h); production default (up to 120 h) cannot respond within 48-hour windows |
| Real-data overfit gate | NSE −1869 → 0.365 | Matches multi-day envelope only; misses diurnal snowmelt entirely; learned P_eff ≈ 0 (coasts on initial storage) |
| Snow, water-year v1 | **INVALID** | Timezone bug: UTC-aware flow index vs. naive SNOTEL index → all-NaN reindex → zero SWE supervision. Result discarded |
| Snow, water-year v2 | NSE 0.659 (vs 0.207 unsupervised) | SWE supervision tripled performance — validated EXP-HYDRO direction |
| Forecast round 1 | Collapsed to assimilation + recession | Losses computed on raw mm/hr (O(0.01)) put NSE denominators at ~1e-6 → ill-conditioned gradients |
| Forecast round 2 (flow scaling) | Plateau ~3.2 | Same plateau despite fix |
| Forecast round 3 (+snow assimilation, 20 ep) | Plateau; P_eff flat at 0.031 mm/hr | Model ignores meteorology |
| **Ceiling probe** (single window) | **NSE 0.610** | Capacity exists — the plateau is an optimization/data problem, not a structural one |
| Forecast round 4 (variance-weighted sampling) | Worse than persistence out-of-sample | Dynamic but wrong; overfits hard training windows |

**Controls run at Isaac's request** (same 372 windows, same split, same loss, same metric):

- GRU seq2seq (~90k params): median window NSE −6.2 / −8.0 / −4.1 — far worse than persistence
  *and* worse than the hybrid's recession mode.
- DA-RNN (daily one-step + 14-day autoregressive rollout): −44 / −39 / −63 — fits training well
  (MSE 0.80 → 0.04) then collapses on rollout (exposure bias).

**Conclusion (triple-confirmed by independent architectures):** 372 windows on one basin cannot
train a ~300k-parameter conditioned model. Notably, even persistence has negative median window
NSE at all leads — 14-day snowmelt forecasting is intrinsically hard. The hybrid's physics prior
degraded most gracefully of the three.

### 4.3 Recurring physics failure mode

With a flow-only loss, the parameter head reliably drives `Tmax` to its lower bound and `Df` to
its upper bound — i.e. "melt the snowpack instantly." Snow is an obstacle between precipitation
and flow unless SWE is directly supervised. This is an equifinality problem, and it recurred
across multiple independent attempts.

### 4.4 Multi-basin — CURRENT PHASE

| Run | Config | Gauged day 1–3 | Ungauged day 1–3 | Note |
|---|---|---|---|---|
| `smoke10` | 10 basins, 8 ep | −99.8% | −0.3% | Evaluated *final* epoch due to EarlyStopper bug |
| `smoke10b` | same, after ES fix | −62.5% | −0.3% | Best-checkpoint restored |
| `smoke10swe` | same + SNODAS | **−28.4%** | **+0.5%** | A/B: only variable changed is snow seeding |
| `fleet_v1_swe` | 99 basins | **+46.7%** | **+2.2%** | **Epoch-0 checkpoint — essentially untrained** |
| `fleet_v2b/v2c/v3` | 99 basins | — | — | Crashed on non-finite gradients |
| `fleet_v5_scaled` | 99 basins, scaling fix | *in progress* | *in progress* | First run that can actually train |

**SNODAS A/B, all bands (smoke, 10 basins, identical configs):**

| Split / band | Baseline | +SNODAS |
|---|---|---|
| Gauged day 1–3 | −62.5% | −28.4% |
| Gauged day 4–7 | −70.1% | −12.9% |
| Gauged day 8–14 | −15.8% | **+18.5%** |
| Gauged all | −27.9% | **+10.4%** |
| Ungauged day 1–3 | −0.3% | +0.5% |
| Ungauged day 8–14 | −26.8% | **−31.4%** ← *worse* |
| Ungauged all | −14.2% | −15.8% ← *worse* |

Snow seeding improves gauged skill at every band and short-lead ungauged skill, but makes
**ungauged long-lead skill worse**. Not investigated. Hypothesis: learned per-basin contexts do
not transfer snow behavior to unseen basins.

---

## 5. Bugs found and fixed (all in FF, all pre-existing unless noted)

1. **`EarlyStopper.check_loss` best-score drift** (`flood_forecast/training_utils.py`) — the
   comparison branch overwrote `best_score` with *worse* scores, so the bar slid downward, early
   stopping never fired, and `checkpoint.pth` retained mediocre weights. `smoke10` was evaluated
   at validation loss 0.886 when its epoch-2 best was 0.326 (2.7× better). **This invalidates any
   earlier FF run that relied on early stopping.**
2. **No best-checkpoint restore on max-epoch completion** (`pytorch_training.py`) — models that
   ran to `max_epochs` without triggering early stopping were evaluated at final-epoch weights.
3. **NaN check ordered after `backward()`/`step()`** (`pytorch_training.py`) — a non-finite loss
   was applied to the weights before being detected.
4. **`clip_grad_norm_` with a non-finite norm** (`pytorch_training.py`) — clipping *scales* by
   the total norm, so a non-finite norm silently poisoned every parameter on `opt.step()`.
5. **DA-RNN `out_feats > 1` decoder wiring** (`flood_forecast/da_rnn/model.py`) — the forward
   pass hard-coded the single-target column split while layers were sized on `out_feats`.
   Fixed; `out_feats=1` verified bit-identical across four config variants. Also fixed a latent
   CUDA device bug (`context` allocated without `.to(device)`).
6. **`CSVDataLoader` unconditional `temp_df.csv` write** — debug artifact written on every
   instantiation.
7. **Unscaled physical channels fed to the Crossformer** (`ode/physics/forecast_training.py`) —
   **this is the significant one.** `met_indices` excluded the flow and SWE columns but not the
   raw physics duplicates: `temp_lapse_k` (256–305 Kelvin) and `sw_raw` (0–1000 W/m²) were fed
   to the transformer alongside z-scored channels of O(1). Diagnosis: on a saved reproducer,
   the forward pass was finite (loss 9.68) but 60 parameters had non-finite gradients — *all* of
   them inside `forcing_generator.encoder.*`, while every other module had max gradient
   magnitude 0.10. The raw channels were also redundant (their scaled counterparts were already
   present). Fix: exclude `raw_temp_index`/`raw_sw_index` from `met_indices`. Verified: zero
   non-finite gradients on the previously-failing batch. **Affects single-basin
   `HybridGR4Forecast` identically.** Reproducer saved at `nonfinite_batch_debug.pth`.

---

## 6. Open issues and unresolved risks

**Ranked by how much they threaten the project's conclusions.**

### 6.1 No demonstrated learning (CRITICAL)

Every positive fleet-scale number to date comes from a checkpoint that had barely trained.
Predicted-vs-actual plots show smooth drift, not hydrograph tracking. Until a properly trained
run produces plots with storm-pulse timing and diurnal melt structure, the correct summary is
"physics + assimilation works; the neural components are unproven."

### 6.2 Pooled vs. median metric divergence (CRITICAL for interpretation)

`fleet_v1_swe` day 1–3 skill distribution across basins:

| Split | n | min | p25 | **median** | p75 | max | frac > 0 |
|---|---|---|---|---|---|---|---|
| Gauged | 96 | −1518% | −41.5% | **−0.1%** | +4.1% | +84.4% | 0.50 |
| Ungauged | 15 | −1090% | −5.8% | **−0.6%** | +1.7% | +4.2% | 0.47 |

The pooled +46.7% is an MSE-ratio over concatenated windows, so it is dominated by
high-variance basins where absolute error reduction is large. **The median basin is at
persistence parity.** Median window NSE is negative at every band (−3.2 to −13.9). Both
statistics are defensible; reporting only the pooled one would be misleading.

### 6.3 Heavy-tailed precipitation inputs (likely secondary instability)

Even after z-scoring, `precipitation` reaches 52σ and `p01m` reaches 92σ, because precipitation
is mostly zeros with rare spikes, giving a tiny standard deviation. This is correct z-scoring of
a pathological distribution, not a bug, but it is poor transformer input. A `log1p` transform
before scaling is the principled fix. Deliberately **not** applied yet, to avoid changing two
things at once while validating the §5.7 fix.

### 6.4 Non-finite-gradient guard introduces sampling bias

When the guard skips batches, those batches are not random — they are likely the most extreme
flow events, i.e. exactly the events of interest. The allowance is now proportional (5% of an
epoch, floor 20). Any run that reports skips should be treated as having a known bias. If §5.7
fully resolves the instability this is moot; `fleet_v5_scaled` has recorded **zero** skips so far.

### 6.5 Multi-basin loader bypasses FF's native evaluation path

`MultiBasinWindowLoader` does not implement the `CSVTestLoader` interface, so `trainer.py`'s
`evaluate_model` — and its automatic W&B prediction plots — never run. `evaluate.py` +
`forecast_report` is the replacement and now logs plots explicitly. A proper test-loader adapter
is outstanding work; until then the two evaluation paths are not cross-validated against each
other, which is itself an audit risk.

### 6.6 Ungauged long-lead regression with SNODAS

See §4.4 — snow seeding made ungauged day 8–14 skill *worse* (−26.8% → −31.4%). Unexplained.

### 6.7 Fixed-step RK4 near thresholds

The snow module's smoothed step functions (`tanh` steepness 5.0) create stiffness. A
semi-implicit solver was identified as the fallback if melt-season integration diverges. The
gradient explosion originally looked like this failure mode and was misdiagnosed as such for
several hours before the real cause (§5.7) was found — a cautionary note for anyone reading
earlier logs.

### 6.8 Environment / operational

- CPU-only machine (MPS available but unused for these runs); long runs go detached under
  `nohup` + `caffeinate`.
- A config key named `sweep` — even set to `false` — force-enables W&B
  (`time_model.wandb_init`).
- FF does **not** auto-discover tests; new test files must be added to `.circleci/config.yml`
  manually. (The `Water` repo does auto-discover.)
- GCS rule: writes only under the `claude_data/` prefix; nothing is ever deleted.

---

## 7. Metric definitions (so the auditor can check them)

- **skill vs. persistence %** = `100 × (1 − MSE_model / MSE_persistence)`, computed on pooled
  windows in mm/hr. 0 = ties persistence. This is the headline metric. Persistence = the
  observed flow at issue time, held flat across the horizon.
- **Bands**: day 1–3 = hours 0–72, day 4–7 = 72–168, day 8–14 = 168–336, all = 0–336.
- **median window NSE**: per-window NSE against that window's own mean. Reported as *secondary
  only* — per-window NSE against the window mean is an oracle baseline and explodes on
  low-variance (winter) windows. An earlier round of this project drew a wrong conclusion from
  pooled NSE alone (persistence scored 0.78 pooled day 1–3 with zero actual skill).
- **Leakage controls**: the loader zeroes target flow across the horizon segment of the source
  window; the SWE channel is excluded from the forcing generator's inputs so future SWE cannot
  reach the horizon simulation. Both are unit-tested (`tests/test_multi_basin_forecast.py`).

---

## 8. Suggested focus for the audit

1. **Verify the leakage controls independently.** `MultiBasinWindowLoader.__getitem__` and
   `HybridGR4Forecast._forecast` / `forward`. Assimilation legitimately uses observed flow at
   t0; confirm nothing beyond t0 reaches the model.
2. **Re-derive the skill metric** from `eval_*/per_basin_metrics.json` and check the pooled vs.
   median story in §6.2 independently.
3. **Check whether the model has actually fitted** — the plots, not the metrics. W&B keys
   `<split>/<site>/forecast_NN_<date>`.
4. **Review the EarlyStopper fix** (§5.1) and consider which historical runs it invalidates.
5. **Confirm the scaling fix is complete** — are any other unscaled physical quantities reaching
   a learned module anywhere in the stack? This class of bug was present for weeks undetected.
6. **Assess the ungauged split's honesty** — 15 basins is small, and they were selected as
   embedded basins with sufficient 2023+ data, which is not a random sample.

---

## 9. Reproduction

```bash
# Build manifest (reads the Water repo's CO scrapes + SNODAS series)
python experiments/catchment_foundation/build_manifest.py \
    --out experiments/catchment_foundation/manifests/co_manifest.json

# Smoke run (10 basins)
python experiments/catchment_foundation/run_training.py --name smoke --max-basins 10 \
    --epochs 15 --samples-per-epoch 1024 --swe

# Full fleet
python experiments/catchment_foundation/run_training.py --name fleet --swe \
    --epochs 30 --samples-per-epoch 16384 --lr 3e-3 --patience 5
```

W&B project `catchment-foundation` (entity `igodfried`) holds every run discussed here.
Branch under audit: `foundation_model_hydro` (7 commits ahead of `master`; 31 files,
+7,513 / −344).
