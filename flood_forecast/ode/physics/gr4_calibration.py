"""
Classical per-basin calibration of the snow-extended GR4 dynamics.

This is the first stage of the dPL-style warm start: a cheap random-search + local-refinement
calibration (a simplified DDS) runs the rigid :class:`~flood_forecast.ode.physics.hydrology.
GR4SnowDynamics` over a basin's training-period forcing and finds parameters that fit observed
flow. The calibrated parameters then become supervised regression targets for the shared
GR4 parameter head, replacing its sigmoid-midpoint initialization with basin-differentiated,
hydrologically plausible values before any end-to-end training.

Candidates are simulated in a single batched pass (the batch dimension is the candidate pool),
so one basin-year costs one fixed-step integration regardless of pool size.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from flood_forecast.ode.physics.hydrology import GR4SnowDynamics

PARAMETER_NAMES = ("X1", "X2", "X3", "X4", "Df", "Tmax", "Tmin")

# Sampling bounds match GR4SnowParameterHead's emission ranges so every calibrated target is
# reachable by the head. "delta" is Tmax - Tmin, mirroring the head's ordering constraint.
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "X1": (10.0, 2000.0), "X2": (-10.0, 10.0), "X3": (5.0, 500.0), "X4": (2.0, 120.0),
    "Df": (0.0, 0.5), "Tmax": (-2.0, 3.0), "delta": (0.0, 4.0),
}
# Store capacities and routing times are scale parameters; sampling them log-uniformly puts the
# prior median near typical literature calibrations (X1 ~ 140 mm, X4 ~ 15 h) instead of the
# bound midpoints that linear sampling (and sigmoid-midpoint hypernetwork init) would give.
LOG_UNIFORM = ("X1", "X3", "X4")


def unit_to_parameters(unit: torch.Tensor,
                       bounds: Dict[str, Tuple[float, float]] = None) -> torch.Tensor:
    """
    Maps unit-cube samples to physical (X1, X2, X3, X4, Df, Tmax, Tmin) parameters.

    :param unit: Samples of shape (n_candidates, 7) in [0, 1], ordered as
        (X1, X2, X3, X4, Df, Tmax, delta).
    :type unit: torch.Tensor
    :param bounds: Per-parameter (lower, upper) bounds, defaults to :data:`DEFAULT_BOUNDS`.
    :type bounds: Dict[str, Tuple[float, float]], optional
    :return: Physical parameters of shape (n_candidates, 7) with Tmin = Tmax - delta.
    :rtype: torch.Tensor
    """
    bounds = bounds or DEFAULT_BOUNDS
    columns = []
    for column, name in enumerate(("X1", "X2", "X3", "X4", "Df", "Tmax", "delta")):
        lower, upper = bounds[name]
        u = unit[:, column]
        if name in LOG_UNIFORM:
            value = torch.exp(np.log(lower) + u * (np.log(upper) - np.log(lower)))
        else:
            value = lower + u * (upper - lower)
        columns.append(value)
    x1, x2, x3, x4, df, tmax, delta = columns
    return torch.stack([x1, x2, x3, x4, df, tmax, tmax - delta], dim=-1)


def simulate_gr4_snow(params: torch.Tensor, forcing: torch.Tensor, substeps: int = 2,
                      production_fill: float = 0.3, routing_fill: float = 0.2) -> torch.Tensor:
    """
    Runs a batched fixed-step (Euler substep) simulation of :class:`GR4SnowDynamics`.

    :param params: Physical parameters of shape (n_candidates, 7).
    :type params: torch.Tensor
    :param forcing: Hourly forcing of shape (n_steps, 4) with channels [P mm/hr, PET mm/hr,
        T degC, SW]; held constant within each hour.
    :type forcing: torch.Tensor
    :param substeps: Euler substeps per hour, defaults to 2.
    :type substeps: int, optional
    :param production_fill: Initial production store as a fraction of X1, defaults to 0.3.
    :type production_fill: float, optional
    :param routing_fill: Initial routing store as a fraction of X3, defaults to 0.2.
    :type routing_fill: float, optional
    :return: Simulated streamflow in mm/hr of shape (n_candidates, n_steps).
    :rtype: torch.Tensor
    """
    dynamics = GR4SnowDynamics(learnable=False)
    dynamics.set_parameters(params)
    n_candidates = params.shape[0]
    state = torch.zeros(n_candidates, dynamics.state_dim, dtype=params.dtype,
                        device=params.device)
    state[:, 1] = production_fill * params[:, 0]
    state[:, 2] = routing_fill * params[:, 2]
    dt = 1.0 / substeps
    flows = torch.empty(n_candidates, forcing.shape[0], dtype=params.dtype,
                        device=params.device)
    for step in range(forcing.shape[0]):
        forcing_step = forcing[step].unsqueeze(0)
        for _ in range(substeps):
            snowfall, rainfall, melt = dynamics.snow_fluxes(forcing_step, state[:, 0])
            liquid = torch.stack(
                [(rainfall + melt).expand(n_candidates), forcing_step[:, 1].expand(n_candidates)],
                dim=-1)
            derivative = torch.cat(
                [(snowfall - melt).expand(n_candidates).unsqueeze(-1),
                 dynamics._derivative(liquid, state[:, 1:])], dim=-1)
            state = (state + dt * derivative).clamp(min=0.0)
        flows[:, step] = dynamics.streamflow(state)
    return flows


def nse(simulated: torch.Tensor, observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes per-candidate Nash-Sutcliffe efficiency against one observed series.

    :param simulated: Simulated flow of shape (n_candidates, n_steps).
    :type simulated: torch.Tensor
    :param observed: Observed flow of shape (n_steps,).
    :type observed: torch.Tensor
    :param mask: Boolean validity mask of shape (n_steps,), defaults to None (all valid).
    :type mask: torch.Tensor, optional
    :return: NSE of shape (n_candidates,).
    :rtype: torch.Tensor
    """
    if mask is None:
        mask = torch.isfinite(observed)
    obs = observed[mask]
    sim = simulated[:, mask]
    residual = ((sim - obs) ** 2).sum(dim=1)
    variance = ((obs - obs.mean()) ** 2).sum().clamp(min=1e-12)
    return 1.0 - residual / variance


def calibrate_gr4_snow(forcing: torch.Tensor, observed: torch.Tensor,
                       mask: Optional[torch.Tensor] = None, n_random: int = 512,
                       chunk_size: int = 128, refine_rounds: int = 3, refine_top: int = 16,
                       refine_samples: int = 128, warmup_hours: int = 2160,
                       substeps: int = 2, seed: int = 42,
                       bounds: Dict[str, Tuple[float, float]] = None) -> Dict:
    """
    Calibrates GR4-snow parameters for one basin by random search plus local refinement.

    :param forcing: Hourly forcing of shape (n_steps, 4): [P mm/hr, PET mm/hr, T degC, SW].
    :type forcing: torch.Tensor
    :param observed: Observed flow in mm/hr of shape (n_steps,); NaNs allowed.
    :type observed: torch.Tensor
    :param mask: Extra validity mask combined with finiteness of ``observed``, defaults to None.
    :type mask: torch.Tensor, optional
    :param n_random: Size of the initial random pool, defaults to 512.
    :type n_random: int, optional
    :param chunk_size: Candidates simulated per batch (memory bound), defaults to 128.
    :type chunk_size: int, optional
    :param refine_rounds: Local refinement rounds after random search, defaults to 3.
    :type refine_rounds: int, optional
    :param refine_top: Elite pool size carried between refinement rounds, defaults to 16.
    :type refine_top: int, optional
    :param refine_samples: Perturbed candidates per refinement round, defaults to 128.
    :type refine_samples: int, optional
    :param warmup_hours: Leading hours excluded from the objective while stores equilibrate,
        defaults to 2160 (90 days).
    :type warmup_hours: int, optional
    :param substeps: Euler substeps per hour, defaults to 2.
    :type substeps: int, optional
    :param seed: Seed for the candidate sampler, defaults to 42.
    :type seed: int, optional
    :param bounds: Sampling bounds, defaults to :data:`DEFAULT_BOUNDS`.
    :type bounds: Dict[str, Tuple[float, float]], optional
    :return: Dict with ``parameters`` (name -> value), ``nse``, ``midpoint_nse`` (the
        sigmoid-midpoint default parameters scored on the same objective) and ``n_valid_hours``.
    :rtype: Dict
    """
    if warmup_hours >= forcing.shape[0]:
        raise ValueError("warmup_hours must be shorter than the forcing record")
    valid = torch.isfinite(observed)
    if mask is not None:
        valid = valid & mask
    valid[:warmup_hours] = False
    if valid.sum() < 24 * 30:
        raise ValueError("Fewer than 30 valid post-warmup days of observed flow")
    generator = torch.Generator().manual_seed(seed)

    def score(unit: torch.Tensor) -> torch.Tensor:
        scores = []
        for start in range(0, unit.shape[0], chunk_size):
            params = unit_to_parameters(unit[start:start + chunk_size], bounds)
            flows = simulate_gr4_snow(params, forcing, substeps=substeps)
            scores.append(nse(flows, observed, valid))
        return torch.cat(scores)

    unit = torch.rand(n_random, 7, generator=generator)
    scores = score(unit)
    for round_index in range(refine_rounds):
        elite_index = scores.argsort(descending=True)[:refine_top]
        elite = unit[elite_index]
        sigma = 0.1 / (2 ** round_index)
        parents = elite[torch.randint(refine_top, (refine_samples,), generator=generator)]
        children = (parents + sigma * torch.randn(refine_samples, 7,
                                                  generator=generator)).clamp(0.0, 1.0)
        unit = torch.cat([elite, children])
        scores = torch.cat([scores[elite_index], score(children)])
    best = scores.argmax()
    best_params = unit_to_parameters(unit[best].unsqueeze(0), bounds)[0]
    midpoint = unit_to_parameters(torch.full((1, 7), 0.5), bounds)
    midpoint_nse = nse(simulate_gr4_snow(midpoint, forcing, substeps=substeps),
                       observed, valid)[0]
    return {
        "parameters": {name: float(best_params[i]) for i, name in enumerate(PARAMETER_NAMES)},
        "nse": float(scores[best]),
        "midpoint_nse": float(midpoint_nse),
        "n_valid_hours": int(valid.sum()),
    }
