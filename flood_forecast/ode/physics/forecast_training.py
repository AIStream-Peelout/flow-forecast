"""
Training and state-initialization for the operational task: hourly flow forecast, 14 days out.

The forecast recipe: (1) run the hybrid model over a spin-up window of observed meteorology to
estimate the ODE state at issue time t0 (under no_grad — the spin-up is state estimation, not a
training signal); (2) correct the runoff-producing states so simulated flow at t0 matches the
*observed* current flow (the "given current flow" input of nowcasting); (3) integrate the forecast
horizon with (forecast) meteorology and train on the horizon flow error.
"""
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from flood_forecast.custom.custom_opt import NSELoss
from flood_forecast.ode.physics.hydrology import HybridGR4Model


def match_current_flow(model: HybridGR4Model, state: torch.Tensor, parameters: torch.Tensor,
                       observed_flow: torch.Tensor, iterations: int = 30) -> torch.Tensor:
    """
    Rescales the runoff-producing states so simulated streamflow equals the observed flow at t0.

    A scalar multiplier per sample is applied to the routing store and Nash cascade states (snow and
    production stores are left untouched — they are supply, not discharge). Streamflow is monotone in
    that multiplier, so a short bisection suffices. Runs under no_grad: this is data assimilation of
    the current observation, not a differentiable operation.

    :param model: The hybrid model (provides dynamics.streamflow and the state layout).
    :type model: HybridGR4Model
    :param state: The spin-up final state of shape (batch_size, state_dim).
    :type state: torch.Tensor
    :param parameters: The active parameters (already set on the dynamics).
    :type parameters: torch.Tensor
    :param observed_flow: Observed flow at t0 in mm/hr of shape (batch_size,).
    :type observed_flow: torch.Tensor
    :param iterations: Bisection iterations, defaults to 30.
    :type iterations: int, optional
    :return: The corrected state of the same shape.
    :rtype: torch.Tensor
    """
    offset = state.shape[1] - (2 + model.dynamics.n_routing_reservoirs) + 1  # first runoff state
    with torch.no_grad():
        low = torch.zeros(state.shape[0], 1, device=state.device)
        high = torch.full_like(low, 1.0)
        # Grow the upper bracket until simulated flow exceeds the observation everywhere.
        for _ in range(40):
            candidate = state.clone()
            candidate[:, offset:] = state[:, offset:] * high
            flow = model.dynamics.streamflow(candidate)
            if (flow >= observed_flow).all():
                break
            high = torch.where((flow < observed_flow).unsqueeze(-1), high * 2.0, high)
        for _ in range(iterations):
            mid = (low + high) / 2.0
            candidate = state.clone()
            candidate[:, offset:] = state[:, offset:] * mid
            flow = model.dynamics.streamflow(candidate)
            below = (flow < observed_flow).unsqueeze(-1)
            low = torch.where(below, mid, low)
            high = torch.where(below, high, mid)
        corrected = state.clone()
        corrected[:, offset:] = state[:, offset:] * (low + high) / 2.0
    return corrected


def estimate_initial_state(model: HybridGR4Model, batch: Dict[str, torch.Tensor],
                           context: torch.Tensor, match_flow: bool = True,
                           spinup_start_state: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Estimates the ODE state at forecast issue time from the spin-up window.

    :param model: The hybrid model.
    :type model: HybridGR4Model
    :param batch: A ForecastWindowDataset batch (needs spinup_met, spinup_raw, spinup_flow).
    :type batch: Dict[str, torch.Tensor]
    :param context: Catchment embeddings of shape (batch_size, context_dim).
    :type context: torch.Tensor
    :param match_flow: Whether to correct the runoff states to the observed flow at t0,
        defaults to True.
    :type match_flow: bool, optional
    :param spinup_start_state: Optional state at the *start* of the spin-up window (e.g. with the
        snow store seeded from SNOTEL/gridded SWE — a 30-day spin-up cannot build a seasonal
        snowpack on its own), defaults to None which uses the model's default initialization.
    :type spinup_start_state: torch.Tensor, optional
    :return: The state at t0 of shape (batch_size, state_dim), detached.
    :rtype: torch.Tensor
    """
    with torch.no_grad():
        raw = batch["spinup_raw"] if model.snow else None
        out = model(batch["spinup_met"], context, initial_state=spinup_start_state,
                    raw_forcing=raw)
        state = out["states"][:, -1, :].clamp(min=0.0)
        if match_flow:
            state = match_current_flow(model, state, out["parameters"],
                                       batch["spinup_flow"][:, -1])
    return state


def train_forecast_model(model: HybridGR4Model, dataset, context: torch.Tensor, epochs: int = 10,
                         batch_size: int = 8, lr: float = 3e-3, device: str = "cpu",
                         match_flow: bool = True, flow_scale: Optional[float] = None,
                         variance_weighted_sampling: bool = True,
                         checkpoint_path: Optional[str] = None, wandb_run=None) -> List[float]:
    """
    Trains the hybrid model on sliding (spin-up -> 14-day horizon) forecast windows.

    :param model: The hybrid model to train.
    :type model: HybridGR4Model
    :param dataset: A ForecastWindowDataset (or any dataset yielding its batch keys).
    :type dataset: torch.utils.data.Dataset
    :param context: The gauge's catchment embedding of shape (1, context_dim); expanded per batch.
    :type context: torch.Tensor
    :param epochs: Training epochs, defaults to 10.
    :type epochs: int, optional
    :param batch_size: Windows per batch, defaults to 8.
    :type batch_size: int, optional
    :param lr: Adam learning rate, defaults to 3e-3.
    :type lr: float, optional
    :param device: Torch device string, defaults to "cpu".
    :type device: str, optional
    :param match_flow: Whether to assimilate the observed flow at t0, defaults to True.
    :type match_flow: bool, optional
    :param flow_scale: Divisor applied to simulated and observed flow inside the loss (use the
        training-set flow standard deviation; basin-normalized losses are essential when flows
        are O(0.01) mm/hr or when training across many basins). Defaults to None which computes
        the dataset's flow standard deviation when available.
    :type flow_scale: float, optional
    :param variance_weighted_sampling: Whether to oversample training windows by their horizon
        flow variance — with equal weighting, the abundant low-variance (recession-like) windows
        dominate the average gradient and push the model toward a no-response solution.
        Defaults to True.
    :type variance_weighted_sampling: bool, optional
    :param checkpoint_path: Where to save the trained state dict, defaults to None.
    :type checkpoint_path: str, optional
    :param wandb_run: An active wandb run; per-epoch losses are logged to it (project policy:
        all experiment training and testing is logged to Weights & Biases), defaults to None.
    :type wandb_run: wandb.sdk.wandb_run.Run, optional
    :return: Mean loss per epoch.
    :rtype: List[float]
    """
    import numpy as np
    model = model.to(device).train()
    context = context.to(device)
    if flow_scale is None:
        flow_scale = float(np.nanstd(getattr(dataset, "flow_mm", np.array([1.0])))) or 1.0
    sampler = None
    if variance_weighted_sampling and hasattr(dataset, "starts"):
        variances = np.array([np.nanvar(dataset.flow_mm[s + dataset.spinup_hours:
                                                        s + dataset.spinup_hours +
                                                        dataset.horizon_hours])
                              for s in dataset.starts])
        weights = np.maximum(variances / max(variances.mean(), 1e-12), 0.1)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(weights).double(), num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None,
                        sampler=sampler, drop_last=len(dataset) > batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    nse = NSELoss()
    mse = torch.nn.MSELoss()
    epoch_losses: List[float] = []
    for epoch in range(epochs):
        total, batches = 0.0, 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            ctx = context.expand(batch["horizon_met"].shape[0], -1)
            initial_state = estimate_initial_state(model, batch, ctx, match_flow=match_flow)
            optimizer.zero_grad()
            raw = batch["horizon_raw"] if model.snow else None
            out = model(batch["horizon_met"], ctx, initial_state=initial_state, raw_forcing=raw)
            sim_scaled = out["flow"] / flow_scale
            obs_scaled = batch["horizon_flow"] / flow_scale
            loss = mse(sim_scaled, obs_scaled) + nse(sim_scaled, obs_scaled)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        epoch_losses.append(total / max(batches, 1))
        print("epoch %d/%d forecast loss %.4f" % (epoch + 1, epochs, epoch_losses[-1]))
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch + 1, "forecast_loss": epoch_losses[-1]})
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
    return epoch_losses


def forecast_report(sim: torch.Tensor, obs: torch.Tensor, persist: torch.Tensor,
                    timestamps, t0s: torch.Tensor, full_flow=None, area_km2: float = None,
                    out_dir: str = None, wandb_run=None, n_examples: int = 6) -> Dict:
    """
    Builds the standard forecast evaluation report: plain-language metrics and per-window plots.

    Metrics per lead band: vanilla MSE (mm/hr squared), RMSE in cfs when the basin area is given,
    median per-window NSE, and the headline "skill vs persistence" (percent MSE reduction relative
    to carrying the current observation forward; 0 = persistence, positive = real skill). Example
    windows (the largest observed event plus a seasonal spread) are rendered with
    :func:`flood_forecast.plot_functions.plot_df_test_with_confidence_interval` showing three days
    of observed history, the issue line, and predicted vs. actual over the horizon.

    :param sim: Simulated horizon flows of shape (n_windows, horizon) in mm/hr.
    :type sim: torch.Tensor
    :param obs: Observed horizon flows of the same shape.
    :type obs: torch.Tensor
    :param persist: Persistence forecasts of the same shape.
    :type persist: torch.Tensor
    :param timestamps: The dataset's hourly DatetimeIndex.
    :type timestamps: pd.DatetimeIndex
    :param t0s: Issue-time positions of each window in timestamps.
    :type t0s: torch.Tensor
    :param full_flow: The dataset's full hourly flow array (for plotting pre-issue history),
        defaults to None which skips the history segment.
    :type full_flow: np.ndarray, optional
    :param area_km2: Basin area for the cfs conversion, defaults to None.
    :type area_km2: float, optional
    :param out_dir: Directory for HTML plots and the metrics JSON, defaults to None (no files).
    :type out_dir: str, optional
    :param wandb_run: An active wandb run to log metrics and figures to, defaults to None.
    :type wandb_run: wandb.sdk.wandb_run.Run, optional
    :param n_examples: Number of example windows to plot, defaults to 6.
    :type n_examples: int, optional
    :return: The metrics dict.
    :rtype: Dict
    """
    import json as _json
    import os as _os

    import numpy as _np
    import pandas as _pd

    from flood_forecast.plot_functions import plot_df_test_with_confidence_interval

    sim_np, obs_np, persist_np = sim.numpy(), obs.numpy(), persist.numpy()
    horizon = sim_np.shape[1]
    bands = {"day1-3": (0, 72), "day4-7": (72, 168), "day8-14": (168, horizon),
             "all": (0, horizon)}
    to_cfs = None if area_km2 is None else area_km2 / (0.0283168 * 3.6)

    def _median_wnse(pred, a, b):
        num = ((pred[:, a:b] - obs_np[:, a:b]) ** 2).sum(1)
        den = _np.maximum(((obs_np[:, a:b] - obs_np[:, a:b].mean(1, keepdims=True)) ** 2).sum(1),
                          1e-8)
        return float(_np.median(1 - num / den))

    metrics: Dict = {}
    for name, (a, b) in bands.items():
        mse_model = float(((sim_np[:, a:b] - obs_np[:, a:b]) ** 2).mean())
        mse_persist = float(((persist_np[:, a:b] - obs_np[:, a:b]) ** 2).mean())
        entry = {"mse_mm_hr2": round(mse_model, 8),
                 "mse_persistence_mm_hr2": round(mse_persist, 8),
                 "skill_vs_persistence_pct": round(100.0 * (1.0 - mse_model / mse_persist), 1),
                 "median_window_nse": round(_median_wnse(sim_np, a, b), 3),
                 "median_window_nse_persistence": round(_median_wnse(persist_np, a, b), 3)}
        if to_cfs is not None:
            entry["rmse_cfs"] = round(float(_np.sqrt(mse_model)) * to_cfs, 1)
            entry["rmse_persistence_cfs"] = round(float(_np.sqrt(mse_persist)) * to_cfs, 1)
        metrics[name] = entry

    # Example windows: the largest observed event plus an even seasonal spread.
    order = list(_np.argsort(-obs_np.max(axis=1)))
    picks = [order[0]] + list(_np.linspace(0, sim_np.shape[0] - 1, n_examples - 1).astype(int))
    figures = {}
    history_hours = 72
    for rank, idx in enumerate(dict.fromkeys(int(i) for i in picks)):
        t0 = int(t0s[idx])
        issue_time = timestamps[t0]
        index = timestamps[t0 - history_hours:t0 + horizon]
        preds = _np.concatenate([_np.full(history_hours, _np.nan), sim_np[idx]])
        if full_flow is not None:
            actual = _np.concatenate([full_flow[t0 - history_hours:t0], obs_np[idx]])
        else:
            actual = _np.concatenate([_np.full(history_hours, _np.nan), obs_np[idx]])
        scale = 1.0 if to_cfs is None else to_cfs
        unit = "mm/hr" if to_cfs is None else "cfs"
        frame = _pd.DataFrame({"preds": preds * scale, "flow_" + unit: actual * scale},
                              index=index)
        figure = plot_df_test_with_confidence_interval(
            frame, _pd.DataFrame({"sample_0": frame["preds"]}), issue_time, {},
            targ_col="flow_" + unit, ci=95.0)
        figure.update_layout(title="Forecast issued %s" % str(issue_time)[:16])
        figures["forecast_%02d_%s" % (rank, str(issue_time)[:10])] = figure

    if out_dir is not None:
        _os.makedirs(out_dir, exist_ok=True)
        with open(_os.path.join(out_dir, "forecast_report.json"), "w") as f:
            _json.dump(metrics, f, indent=2)
        for name, figure in figures.items():
            figure.write_html(_os.path.join(out_dir, name + ".html"))
    if wandb_run is not None:
        import wandb as _wandb
        flat = {band + "/" + key: value for band, entry in metrics.items()
                for key, value in entry.items()}
        wandb_run.log(flat)
        for name, figure in figures.items():
            wandb_run.log({name: _wandb.Plotly(figure)})
    return metrics


class HybridGR4Forecast(torch.nn.Module):
    """
    FF-config-compatible wrapper of the hybrid GR4 forecaster (registered as "HybridGR4").

    Adapts :class:`~flood_forecast.ode.physics.hydrology.HybridGR4Model` to the standard framework
    interface — ``forward(x)`` with x from :class:`CatchmentWindowLoader` of shape
    (batch, spinup + horizon, n_features) — so PyTorchForecast, train_transformer_style, the
    evaluator and W&B logging all work unchanged. Internally: no-grad spin-up over the first
    ``spinup_length`` rows, assimilation of the observed flow at issue time, then the differentiable
    horizon simulation. The catchment context is a learnable embedding by default (single-basin
    configs) or a fixed vector loaded from ``context_path``.

    Feature-order contract (column order of relevant_cols in the config): the target flow column
    first, then met columns, with raw temperature (Kelvin) and shortwave as the channels named by
    ``raw_temp_index``/``raw_sw_index``. Flow and raw channels must be excluded from scaling
    (``scaled_cols``) so the physics receives physical units.
    """

    def __init__(self, n_time_series: int, spinup_length: int, forecast_length: int,
                 raw_temp_index: int, raw_sw_index: int, context_dim: int = 256, dim: int = 64,
                 depth: int = 2, heads: int = 4, snow: bool = True, temp_offset_c: float = 0.0,
                 match_flow: bool = True, context_path: Optional[str] = None,
                 parameter_head_params: Optional[dict] = None):
        """
        Initializes the wrapper.

        :param n_time_series: Total feature count of the loader (target + met + raw channels).
        :type n_time_series: int
        :param spinup_length: The spin-up segment length within each source window.
        :type spinup_length: int
        :param forecast_length: The forecast horizon length.
        :type forecast_length: int
        :param raw_temp_index: Column index of raw temperature in Kelvin.
        :type raw_temp_index: int
        :param raw_sw_index: Column index of raw shortwave radiation.
        :type raw_sw_index: int
        :param context_dim: Catchment embedding dimension, defaults to 256.
        :type context_dim: int, optional
        :param dim: Forcing generator width, defaults to 64.
        :type dim: int, optional
        :param depth: Forcing generator depth, defaults to 2.
        :type depth: int, optional
        :param heads: Attention heads, defaults to 4.
        :type heads: int, optional
        :param snow: Whether to use the snow-extended dynamics, defaults to True.
        :type snow: bool, optional
        :param temp_offset_c: Lapse-rate correction added to the raw temperature channel in degC,
            defaults to 0.0.
        :type temp_offset_c: float, optional
        :param match_flow: Whether to assimilate observed flow at issue time, defaults to True.
        :type match_flow: bool, optional
        :param context_path: Optional .pt file with a fixed context vector; when None the context
            is a learnable parameter, defaults to None.
        :type context_path: str, optional
        :param parameter_head_params: Passed through to the parameter head, defaults to None.
        :type parameter_head_params: dict, optional
        """
        super().__init__()
        self.spinup_length = spinup_length
        self.forecast_length = forecast_length
        self.raw_indices = [raw_temp_index, raw_sw_index]
        self.met_indices = [i for i in range(n_time_series) if i not in (0,)]
        self.temp_offset_c = temp_offset_c
        self.match_flow = match_flow
        self.hybrid = HybridGR4Model(n_met_features=len(self.met_indices),
                                     seq_len=spinup_length, context_dim=context_dim, dim=dim,
                                     depth=depth, heads=heads, snow=snow,
                                     parameter_head_params=parameter_head_params)
        if context_path is not None:
            self.register_buffer("context", torch.load(context_path,
                                                       weights_only=True).reshape(1, -1))
        else:
            self.context = torch.nn.Parameter(0.1 * torch.randn(1, context_dim))

    def _raw(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Extracts the [temperature degC, shortwave] physics channels from a window segment.

        :param segment: A window segment of shape (batch, steps, n_features).
        :type segment: torch.Tensor
        :return: Raw forcing of shape (batch, steps, 2).
        :rtype: torch.Tensor
        """
        temp_c = segment[:, :, self.raw_indices[0]] - 273.15 + self.temp_offset_c
        return torch.stack([temp_c, segment[:, :, self.raw_indices[1]]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs spin-up, assimilation and the horizon simulation.

        :param x: Source windows of shape (batch, spinup + horizon, n_features); column 0 is the
            physical flow (mm/hr, zeroed in the horizon segment by the loader).
        :type x: torch.Tensor
        :return: Simulated horizon flow of shape (batch, forecast_length) in mm/hr.
        :rtype: torch.Tensor
        """
        spin, horizon = x[:, :self.spinup_length], x[:, self.spinup_length:]
        context = self.context.expand(x.shape[0], -1)
        batch = {"spinup_met": spin[:, :, self.met_indices], "spinup_flow": spin[:, :, 0],
                 "spinup_raw": self._raw(spin)}
        initial_state = estimate_initial_state(self.hybrid, batch, context,
                                               match_flow=self.match_flow)
        out = self.hybrid(horizon[:, :, self.met_indices], context, initial_state=initial_state,
                          raw_forcing=self._raw(horizon) if self.hybrid.snow else None)
        return out["flow"]
