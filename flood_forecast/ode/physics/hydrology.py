"""
Hydrology-specific ODE dynamics.

Currently implements the continuous state-space formulation of the GR4 rainfall-runoff model following
Santos et al. (2018), "Continuous state-space representation of a bucket-type rainfall-runoff model: a case
study with the GR4 model using state-space GR4" (Geosci. Model Dev. 11). The discrete unit hydrographs of
the classic GR4J/GR4H are replaced by a Nash cascade of linear reservoirs so the whole model is a smooth,
differentiable ODE suitable for torchdiffeq.

All storages are in mm, forcings in mm per unit time (use hourly rates with hourly integration times for a
GR4H-style setup) and the simulated streamflow is in mm per unit time over the catchment. Conversion to
volumetric discharge (e.g. m^3/s) is an affine transform using the basin area and belongs downstream.
"""
from typing import Optional

import torch

from flood_forecast.ode.dynamics import ForcedDynamics, _inverse_softplus, register_dynamics


class GR4Dynamics(ForcedDynamics):
    """
    Continuous state-space GR4 rainfall-runoff dynamics.

    The state vector is ``[S, R, V_1, ..., V_n]`` where S is the production store, R the routing store and
    V_i the stores of a Nash cascade approximating the GR4 unit hydrograph. The forcing has two channels,
    ``[P, E]``: precipitation and potential evapotranspiration.

    The four GR4 parameters are: X1 the production store capacity (mm), X2 the groundwater exchange
    coefficient (mm per unit time, may be negative), X3 the routing store capacity (mm) and X4 the unit
    hydrograph time constant (in integration time units). They can either be learned globally as module
    parameters (the default, matching the other dynamics classes) or supplied per sample via
    :meth:`set_parameters` — the intended path for a hypernetwork that maps catchment embeddings to
    parameters. Externally supplied parameters must already be constrained (X1, X3, X4 strictly positive).
    """

    forcing_dim = 2

    def __init__(self, x1_init: float = 300.0, x2_init: float = 0.0, x3_init: float = 100.0,
                 x4_init: float = 24.0, n_routing_reservoirs: int = 3, learnable: bool = True,
                 interpolation: str = "previous"):
        """
        Initializes the GR4 dynamics with globally learnable parameters.

        :param x1_init: Initial production store capacity in mm, defaults to 300.0.
        :type x1_init: float, optional
        :param x2_init: Initial groundwater exchange coefficient in mm per unit time, defaults to 0.0.
        :type x2_init: float, optional
        :param x3_init: Initial routing store capacity in mm, defaults to 100.0.
        :type x3_init: float, optional
        :param x4_init: Initial unit hydrograph time constant in integration time units, defaults to 24.0
            (i.e. one day when integrating hourly).
        :type x4_init: float, optional
        :param n_routing_reservoirs: The number of Nash cascade reservoirs approximating the unit
            hydrograph, defaults to 3.
        :type n_routing_reservoirs: int, optional
        :param learnable: Whether the global parameters are trained by gradient descent, defaults to True.
        :type learnable: bool, optional
        :param interpolation: Forcing interpolation mode, "previous" or "linear", defaults to "previous".
        :type interpolation: str, optional
        """
        super().__init__(interpolation=interpolation)
        self.n_routing_reservoirs = n_routing_reservoirs
        self.state_dim = 2 + n_routing_reservoirs
        raw_positive = torch.tensor([_inverse_softplus(x1_init), _inverse_softplus(x3_init),
                                     _inverse_softplus(x4_init)])
        self.raw_x1_x3_x4 = torch.nn.Parameter(raw_positive, requires_grad=learnable)
        self.raw_x2 = torch.nn.Parameter(torch.tensor(x2_init), requires_grad=learnable)
        self._external_params: Optional[torch.Tensor] = None

    def set_parameters(self, params: Optional[torch.Tensor]) -> None:
        """
        Overrides the global GR4 parameters with per-sample values (e.g. from a hypernetwork).

        :param params: A tensor of shape (batch_size, 4) holding (X1, X2, X3, X4) per sample with X1, X3
            and X4 already constrained to be strictly positive, or None to revert to the global learnable
            parameters.
        :type params: torch.Tensor, optional
        :return: None
        :rtype: None
        """
        if params is not None and params.shape[-1] != 4:
            raise ValueError("Expected params of shape (batch_size, 4) but got " + str(list(params.shape)))
        self._external_params = params

    def gr4_parameters(self) -> torch.Tensor:
        """
        Returns the currently active constrained GR4 parameters.

        :return: A tensor of shape (batch_size, 4) if per-sample parameters are set, otherwise of
            shape (1, 4) with the softplus-constrained global parameters.
        :rtype: torch.Tensor
        """
        if self._external_params is not None:
            return self._external_params
        x1, x3, x4 = torch.nn.functional.softplus(self.raw_x1_x3_x4).unbind(-1)
        return torch.stack([x1, self.raw_x2, x3, x4]).reshape(1, 4)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the GR4 state derivatives at solver time t.

        :param t: A scalar tensor with the current integration time.
        :type t: torch.Tensor
        :param state: The current state ``[S, R, V_1..V_n]`` of shape (batch_size, 2 + n_routing_reservoirs).
        :type state: torch.Tensor
        :return: The state derivative of the same shape.
        :rtype: torch.Tensor
        """
        return self._derivative(self.forcing_at(t), state)

    def _derivative(self, forcing: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the GR4 derivative for an explicit liquid forcing (shared with subclasses).

        :param forcing: Forcing of shape (batch_size, >=2) whose first two channels are
            [P_liquid, PET].
        :type forcing: torch.Tensor
        :param state: The GR4 state ``[S, R, V_1..V_n]`` of shape (batch_size, 2 + n_reservoirs).
        :type state: torch.Tensor
        :return: The state derivative of the same shape.
        :rtype: torch.Tensor
        """
        precip, pet = forcing[:, 0].clamp(min=0.0), forcing[:, 1].clamp(min=0.0)
        params = self.gr4_parameters()
        x1, x2, x3, x4 = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        production = state[:, 0]
        routing = state[:, 1]
        cascade = state[:, 2:]

        # Production store: infiltration, actual ET and power-law percolation (Santos et al. 2018, eq. 15).
        fill = (production / x1).clamp(0.0, 1.0)
        infiltration = precip * (1.0 - fill ** 2)
        actual_et = pet * fill * (2.0 - fill)
        percolation = (4.0 / 9.0) ** 4 / 4.0 * x1 * fill ** 5
        d_production = infiltration - actual_et - percolation

        # Nash cascade replacing the unit hydrographs; inflow is the effective rainfall.
        effective_rain = precip * fill ** 2 + percolation
        rate = self.n_routing_reservoirs / x4.unsqueeze(-1)
        outflows = rate * cascade.clamp(min=0.0)
        d_cascade = -outflows
        d_cascade = d_cascade + torch.cat(
            [effective_rain.unsqueeze(-1), outflows[:, :-1]], dim=-1
        )
        routed = outflows[:, -1]

        # Routing store with groundwater exchange; 90/10 split between routed and direct branches.
        routing_fill = (routing / x3).clamp(min=0.0)
        exchange = x2 * routing_fill ** 3.5
        routed_outflow = x3 / 4.0 * routing_fill.clamp(max=1.0) ** 5
        d_routing = 0.9 * routed + exchange - routed_outflow

        return torch.cat([d_production.unsqueeze(-1), d_routing.unsqueeze(-1), d_cascade], dim=-1)

    def streamflow(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the simulated streamflow from a state (an observation function, not part of the state).

        :param state: States of shape (batch_size, state_dim) or (batch_size, n_times, state_dim).
        :type state: torch.Tensor
        :return: Streamflow in mm per unit time of shape (batch_size,) or (batch_size, n_times).
        :rtype: torch.Tensor
        """
        params = self.gr4_parameters()
        x2, x3, x4 = params[..., 1], params[..., 2], params[..., 3]
        if state.dim() == 3:
            x2, x3, x4 = x2.unsqueeze(-1), x3.unsqueeze(-1), x4.unsqueeze(-1)
        routing = state[..., 1]
        routed = self.n_routing_reservoirs / x4 * state[..., -1].clamp(min=0.0)
        routing_fill = (routing / x3).clamp(min=0.0)
        exchange = x2 * routing_fill ** 3.5
        routed_outflow = x3 / 4.0 * routing_fill.clamp(max=1.0) ** 5
        direct_outflow = (0.1 * routed + exchange).clamp(min=0.0)
        return routed_outflow + direct_outflow

    def actual_et(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes actual evapotranspiration at solver time t (for supervision against satellite ET).

        :param t: A scalar tensor with the current integration time.
        :type t: torch.Tensor
        :param state: The current state of shape (batch_size, state_dim).
        :type state: torch.Tensor
        :return: Actual ET in mm per unit time of shape (batch_size,).
        :rtype: torch.Tensor
        """
        pet = self.forcing_at(t)[:, 1].clamp(min=0.0)
        x1 = self.gr4_parameters()[:, 0]
        fill = (state[:, 0] / x1).clamp(0.0, 1.0)
        return pet * fill * (2.0 - fill)


def smooth_step(x: torch.Tensor) -> torch.Tensor:
    """
    The smoothed step function of Höge et al. (2022, HESS 26:5085), verbatim from HydroNODE.

    ``step_fct(x) = (tanh(5.0 * (x - 0.5)) + 1.0) * 0.5`` — note the -0.5 shift (a smoothed step
    centered at x = 0.5) and steepness 5.0; both are part of the published formulation and the
    steepness also bounds ODE stiffness.

    :param x: The argument tensor.
    :type x: torch.Tensor
    :return: A smooth 0-to-1 step of the same shape.
    :rtype: torch.Tensor
    """
    return (torch.tanh(5.0 * (x - 0.5)) + 1.0) * 0.5


class GR4SnowDynamics(GR4Dynamics):
    """
    Continuous GR4 with an EXP-HYDRO snow bucket prepended (Patil & Stieglitz 2014 as smoothed by
    Höge et al. 2022).

    The state vector becomes ``[S_snow, S, R, V_1..V_n]``. Precipitation is partitioned into snowfall
    and rainfall by temperature, snowfall accumulates in the snow store, temperature-indexed melt
    drains it, and the production store receives rainfall + melt instead of raw precipitation — giving
    melt-driven flow the physical (and gradient) pathway plain GR4 lacks.

    Forcing has four channels: ``[P, PET, T]`` plus a spare shortwave channel reserved for an
    enhanced temperature-index melt term (unused by the default degree-day melt). Parameters are the
    four GR4 parameters plus ``Df`` (melt factor, mm/degC per unit time), ``Tmax`` (melt threshold,
    degC) and ``Tmin`` (rain/snow partition temperature, degC), i.e. per-sample tensors of shape
    (batch_size, 7) via :meth:`GR4Dynamics.set_parameters`.
    """

    forcing_dim = 4

    def __init__(self, df_init: float = 0.1, tmax_init: float = 0.5, tmin_init: float = 0.0,
                 **gr4_kwargs):
        """
        Initializes the snow-extended dynamics.

        :param df_init: Initial degree-day melt factor in mm/degC per unit time (hourly: daily
            literature values of 1-10 mm/degC/day divide by 24), defaults to 0.1.
        :type df_init: float, optional
        :param tmax_init: Initial melt threshold temperature in degC, defaults to 0.5.
        :type tmax_init: float, optional
        :param tmin_init: Initial rain/snow partition temperature in degC, defaults to 0.0.
        :type tmin_init: float, optional
        :param gr4_kwargs: Keyword arguments forwarded to :class:`GR4Dynamics`.
        :type gr4_kwargs: dict
        """
        super().__init__(**gr4_kwargs)
        self.state_dim = 3 + self.n_routing_reservoirs
        raw_snow = torch.tensor([_inverse_softplus(df_init), tmax_init, tmin_init])
        self.raw_snow_params = torch.nn.Parameter(raw_snow,
                                                  requires_grad=self.raw_x2.requires_grad)

    def set_parameters(self, params: Optional[torch.Tensor]) -> None:
        """
        Overrides parameters with per-sample values (X1, X2, X3, X4, Df, Tmax, Tmin).

        :param params: A tensor of shape (batch_size, 7) with X1/X3/X4/Df already strictly positive
            and Tmin <= Tmax, or None to revert to the global learnable parameters.
        :type params: torch.Tensor, optional
        :return: None
        :rtype: None
        """
        if params is not None and params.shape[-1] != 7:
            raise ValueError("Expected params of shape (batch_size, 7) but got " +
                             str(list(params.shape)))
        self._external_params = params

    def gr4_parameters(self) -> torch.Tensor:
        """
        Returns the active constrained parameters, GR4 columns first.

        :return: A tensor of shape (batch_size, 7) or (1, 7): (X1, X2, X3, X4, Df, Tmax, Tmin).
        :rtype: torch.Tensor
        """
        if self._external_params is not None:
            return self._external_params
        x1, x3, x4 = torch.nn.functional.softplus(self.raw_x1_x3_x4).unbind(-1)
        df = torch.nn.functional.softplus(self.raw_snow_params[0])
        return torch.stack([x1, self.raw_x2, x3, x4, df, self.raw_snow_params[1],
                            self.raw_snow_params[2]]).reshape(1, 7)

    def snow_fluxes(self, forcing: torch.Tensor,
                    snow_store: torch.Tensor) -> tuple:
        """
        Computes the smoothed snowfall, rainfall and melt fluxes (Höge et al. 2022, verbatim forms).

        :param forcing: The interpolated forcing of shape (batch_size, 4): [P, PET, T, SW].
        :type forcing: torch.Tensor
        :param snow_store: The snow store S_snow of shape (batch_size,).
        :type snow_store: torch.Tensor
        :return: A tuple (snowfall, rainfall, melt), each of shape (batch_size,).
        :rtype: tuple
        """
        params = self.gr4_parameters()
        df, tmax, tmin = params[..., 4], params[..., 5], params[..., 6]
        precip = forcing[:, 0].clamp(min=0.0)
        temperature = forcing[:, 2]
        snowfall = smooth_step(tmin - temperature) * precip
        rainfall = smooth_step(temperature - tmin) * precip
        melt = smooth_step(temperature - tmax) * smooth_step(snow_store) * \
            torch.minimum(snow_store.clamp(min=0.0), df * (temperature - tmax).clamp(min=0.0))
        return snowfall, rainfall, melt

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the snow + GR4 state derivatives at solver time t.

        :param t: A scalar tensor with the current integration time.
        :type t: torch.Tensor
        :param state: The state ``[S_snow, S, R, V_1..V_n]`` of shape (batch_size, state_dim).
        :type state: torch.Tensor
        :return: The state derivative of the same shape.
        :rtype: torch.Tensor
        """
        forcing = self.forcing_at(t)
        snowfall, rainfall, melt = self.snow_fluxes(forcing, state[:, 0])
        d_snow = snowfall - melt
        # GR4 below the snow bucket sees rainfall + melt as its liquid water input.
        liquid_forcing = torch.stack([rainfall + melt, forcing[:, 1]], dim=-1)
        return torch.cat([d_snow.unsqueeze(-1), self._derivative(liquid_forcing, state[:, 1:])],
                         dim=-1)

    def streamflow(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes streamflow from a snow-extended state (indices shifted by the snow store).

        :param state: States of shape (batch_size, state_dim) or (batch_size, n_times, state_dim).
        :type state: torch.Tensor
        :return: Streamflow in mm per unit time.
        :rtype: torch.Tensor
        """
        return super().streamflow(state[..., 1:])

    def swe(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns the snow water equivalent state (for supervision against SNOTEL SWE).

        :param state: States of shape (batch_size, state_dim) or (batch_size, n_times, state_dim).
        :type state: torch.Tensor
        :return: SWE in mm.
        :rtype: torch.Tensor
        """
        return state[..., 0]

    def actual_et(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes actual evapotranspiration (production store sits at index 1 behind the snow store).

        :param t: A scalar tensor with the current integration time.
        :type t: torch.Tensor
        :param state: The current state of shape (batch_size, state_dim).
        :type state: torch.Tensor
        :return: Actual ET in mm per unit time of shape (batch_size,).
        :rtype: torch.Tensor
        """
        pet = self.forcing_at(t)[:, 1].clamp(min=0.0)
        x1 = self.gr4_parameters()[:, 0]
        fill = (state[:, 1] / x1).clamp(0.0, 1.0)
        return pet * fill * (2.0 - fill)


class GR4SnowBandsDynamics(GR4SnowDynamics):
    """
    Elevation-banded snow on top of GR4: one snow store per equal-area elevation band.

    Each band sees a lapse-rate-adjusted temperature, so high bands accumulate and hold snow while
    low bands rain and melt — reproducing the staggered melt-out (and late-season high-elevation
    trickle) that a single lumped bucket cannot. Snow parameters (Df, Tmax, Tmin) are shared across
    bands; the bands differ only in temperature. The state is ``[S_snow_1..S_snow_B, S, R, V_..]``
    with band 1 the lowest. Band geometry (elevation offsets from the forcing temperature's
    reference elevation, and area fractions) is set via :meth:`set_band_geometry` — e.g. from a
    SNOTEL-fitted profile.
    """

    def __init__(self, n_bands: int = 5, lapse_rate: float = -0.0065, **snow_kwargs):
        """
        Initializes the banded snow dynamics.

        :param n_bands: The number of elevation bands, defaults to 5.
        :type n_bands: int, optional
        :param lapse_rate: The temperature lapse rate in degC per meter, defaults to -0.0065.
        :type lapse_rate: float, optional
        :param snow_kwargs: Keyword arguments forwarded to :class:`GR4SnowDynamics`.
        :type snow_kwargs: dict
        """
        super().__init__(**snow_kwargs)
        self.n_bands = n_bands
        self.lapse_rate = lapse_rate
        self.state_dim = n_bands + 2 + self.n_routing_reservoirs
        self.register_buffer("band_offsets_m", torch.zeros(n_bands))
        self.register_buffer("band_fractions", torch.full((n_bands,), 1.0 / n_bands))

    def set_band_geometry(self, elevation_offsets_m, area_fractions) -> None:
        """
        Sets the band elevations (relative to the forcing temperature's reference elevation) and
        area fractions.

        :param elevation_offsets_m: Per-band elevation offsets in meters (positive = higher than the
            reference), lowest band first.
        :type elevation_offsets_m: Sequence[float]
        :param area_fractions: Per-band area fractions summing to 1.
        :type area_fractions: Sequence[float]
        :return: None
        :rtype: None
        """
        self.band_offsets_m = torch.as_tensor(elevation_offsets_m, dtype=torch.float32)
        self.band_fractions = torch.as_tensor(area_fractions, dtype=torch.float32)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes banded-snow + GR4 derivatives at solver time t.

        :param t: A scalar tensor with the current integration time.
        :type t: torch.Tensor
        :param state: The state ``[S_snow_1..S_snow_B, S, R, V..]`` of shape (batch_size, state_dim).
        :type state: torch.Tensor
        :return: The state derivative of the same shape.
        :rtype: torch.Tensor
        """
        forcing = self.forcing_at(t)
        params = self.gr4_parameters()
        df, tmax, tmin = params[..., 4:5], params[..., 5:6], params[..., 6:7]
        precip = forcing[:, 0:1].clamp(min=0.0)
        band_temp = forcing[:, 2:3] + self.lapse_rate * self.band_offsets_m.unsqueeze(0)
        snow_states = state[:, :self.n_bands]

        snowfall = smooth_step(tmin - band_temp) * precip
        rainfall = smooth_step(band_temp - tmin) * precip
        melt = smooth_step(band_temp - tmax) * smooth_step(snow_states) * \
            torch.minimum(snow_states.clamp(min=0.0), df * (band_temp - tmax).clamp(min=0.0))
        d_snow = snowfall - melt

        liquid = ((rainfall + melt) * self.band_fractions.unsqueeze(0)).sum(dim=-1)
        liquid_forcing = torch.stack([liquid, forcing[:, 1]], dim=-1)
        return torch.cat([d_snow, self._derivative(liquid_forcing, state[:, self.n_bands:])],
                         dim=-1)

    def streamflow(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes streamflow from a banded state.

        :param state: States of shape (..., state_dim).
        :type state: torch.Tensor
        :return: Streamflow in mm per unit time.
        :rtype: torch.Tensor
        """
        return GR4Dynamics.streamflow(self, state[..., self.n_bands:])

    def swe(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns the area-weighted basin-mean SWE (per-band states are in ``state[..., :n_bands]``).

        :param state: States of shape (..., state_dim).
        :type state: torch.Tensor
        :return: Basin-mean SWE in mm.
        :rtype: torch.Tensor
        """
        return (state[..., :self.n_bands] * self.band_fractions).sum(dim=-1)

    def actual_et(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes actual ET (production store sits behind the band states).

        :param t: A scalar tensor with the current integration time.
        :type t: torch.Tensor
        :param state: The current state of shape (batch_size, state_dim).
        :type state: torch.Tensor
        :return: Actual ET in mm per unit time of shape (batch_size,).
        :rtype: torch.Tensor
        """
        pet = self.forcing_at(t)[:, 1].clamp(min=0.0)
        x1 = self.gr4_parameters()[:, 0]
        fill = (state[:, self.n_bands] / x1).clamp(0.0, 1.0)
        return pet * fill * (2.0 - fill)


class GR4ParameterHead(torch.nn.Module):
    """
    A hypernetwork head mapping a catchment embedding to the four GR4 parameters.

    Each parameter is squashed by a scaled sigmoid into a physically realistic range, so the ODE can
    never receive degenerate values regardless of the embedding. The output plugs straight into
    :meth:`GR4Dynamics.set_parameters`.
    """

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 64,
                 x1_range: tuple = (10.0, 2000.0), x2_range: tuple = (-10.0, 10.0),
                 x3_range: tuple = (5.0, 500.0), x4_range: tuple = (2.0, 120.0)):
        """
        Initializes the parameter head.

        :param embedding_dim: The catchment embedding dimension, defaults to 256.
        :type embedding_dim: int, optional
        :param hidden_dim: The hidden layer width, defaults to 64.
        :type hidden_dim: int, optional
        :param x1_range: Bounds (mm) for the production store capacity, defaults to (10, 2000).
        :type x1_range: tuple, optional
        :param x2_range: Bounds (mm per unit time) for the exchange coefficient, defaults to (-10, 10).
        :type x2_range: tuple, optional
        :param x3_range: Bounds (mm) for the routing store capacity, defaults to (5, 500).
        :type x3_range: tuple, optional
        :param x4_range: Bounds (time units) for the unit hydrograph constant, defaults to
            (2.0, 120). The lower bound is a SOLVER STABILITY constraint, not just a physical one:
            the Nash cascade drains at rate ``n_routing_reservoirs / X4``, and explicit RK4 is
            stable on the real axis only for ``rate * dt < 2.785``. With the default three
            reservoirs and hourly steps that requires ``X4 > 3 / 2.785 = 1.077``; anything below
            it diverges and produces gradient norms that overflow float32. 2.0 keeps a margin and
            is physically honest — a catchment with a sub-2-hour unit hydrograph is not a
            catchment. Lower this ONLY alongside a smaller solver step or an implicit method.
        :type x4_range: tuple, optional
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim), torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 4),
        )
        # Near-zero-init the output layer so every catchment starts close to mid-range parameters,
        # keeping the scaled sigmoid in its linear region — otherwise early training slams parameters
        # into the bounds where the sigmoid saturates and gradients vanish. The weights stay small but
        # nonzero so the embedding has a nonzero Jacobian into the parameters from the first step.
        torch.nn.init.normal_(self.net[-1].weight, std=1e-3)
        torch.nn.init.zeros_(self.net[-1].bias)
        bounds = torch.tensor([x1_range, x2_range, x3_range, x4_range])
        self.register_buffer("lower", bounds[:, 0])
        self.register_buffer("upper", bounds[:, 1])

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Maps embeddings to bounded GR4 parameters.

        :param embedding: Catchment embeddings of shape (batch_size, embedding_dim).
        :type embedding: torch.Tensor
        :return: Parameters (X1, X2, X3, X4) of shape (batch_size, 4), each within its range.
        :rtype: torch.Tensor
        """
        squashed = torch.sigmoid(self.net(embedding))
        return self.lower + squashed * (self.upper - self.lower)


class GR4SnowParameterHead(GR4ParameterHead):
    """
    Hypernetwork head emitting the seven GR4-with-snow parameters.

    Emits (X1, X2, X3, X4, Df, Tmax, Tmin) with Tmin derived as ``Tmax - delta`` (delta >= 0), so the
    rain/snow partition temperature can never exceed the melt threshold.
    """

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 64,
                 df_range: tuple = (0.0, 0.5), tmax_range: tuple = (-2.0, 3.0),
                 delta_range: tuple = (0.0, 4.0), **gr4_ranges):
        """
        Initializes the snow parameter head.

        :param embedding_dim: The catchment embedding dimension, defaults to 256.
        :type embedding_dim: int, optional
        :param hidden_dim: The hidden layer width, defaults to 64.
        :type hidden_dim: int, optional
        :param df_range: Bounds for the melt factor in mm/degC per unit time (hourly: literature
            daily values 1-10 mm/degC/day divided by 24), defaults to (0, 0.5).
        :type df_range: tuple, optional
        :param tmax_range: Bounds (degC) for the melt threshold, defaults to (-2, 3).
        :type tmax_range: tuple, optional
        :param delta_range: Bounds (degC) for Tmax - Tmin, defaults to (0, 4).
        :type delta_range: tuple, optional
        :param gr4_ranges: The x1_range/x2_range/x3_range/x4_range keyword arguments of
            :class:`GR4ParameterHead`.
        :type gr4_ranges: dict
        """
        super().__init__(embedding_dim=embedding_dim, hidden_dim=hidden_dim, **gr4_ranges)
        self.snow_net = torch.nn.Linear(hidden_dim, 3)
        torch.nn.init.normal_(self.snow_net.weight, std=1e-3)
        torch.nn.init.zeros_(self.snow_net.bias)
        snow_bounds = torch.tensor([df_range, tmax_range, delta_range])
        self.register_buffer("snow_lower", snow_bounds[:, 0])
        self.register_buffer("snow_upper", snow_bounds[:, 1])

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Maps embeddings to the seven bounded parameters.

        :param embedding: Catchment embeddings of shape (batch_size, embedding_dim).
        :type embedding: torch.Tensor
        :return: Parameters (X1, X2, X3, X4, Df, Tmax, Tmin) of shape (batch_size, 7).
        :rtype: torch.Tensor
        """
        hidden = self.net[:-1](embedding)
        gr4 = self.lower + torch.sigmoid(self.net[-1](hidden)) * (self.upper - self.lower)
        snow = self.snow_lower + torch.sigmoid(self.snow_net(hidden)) * \
            (self.snow_upper - self.snow_lower)
        df, tmax, delta = snow.unbind(-1)
        return torch.cat([gr4, df.unsqueeze(-1), tmax.unsqueeze(-1),
                          (tmax - delta).unsqueeze(-1)], dim=-1)


class EffectiveForcingGenerator(torch.nn.Module):
    """
    The dynamic module: maps recent hourly meteorology to effective precipitation and PET.

    A transformer encodes the met window, the catchment embedding is injected at every step through
    :class:`~flood_forecast.meta_models.merging_model.GatedFusion`, and a softplus head guarantees the
    effective forcings are non-negative. Resolves input uncertainty (gauge undercatch, gridded vs.
    point precip disagreement) by letting the network reinterpret the raw forcings per catchment.
    """

    def __init__(self, n_met_features: int, seq_len: int, context_dim: int = 256, dim: int = 64,
                 depth: int = 2, heads: int = 4, dim_head: int = 32, dropout: float = 0.0,
                 encoder_type: str = "crossformer", seg_len: int = 3, anchored: bool = False,
                 use_multiplier: bool = True, use_asos_gate: bool = False):
        """
        Initializes the forcing generator.

        :param n_met_features: The number of raw meteorological input channels.
        :type n_met_features: int
        :param seq_len: The (fixed) met window length in time steps.
        :type seq_len: int
        :param context_dim: The catchment embedding dimension, defaults to 256.
        :type context_dim: int, optional
        :param dim: The sequence encoder embedding dimension, defaults to 64.
        :type dim: int, optional
        :param depth: The number of encoder blocks/scales, defaults to 2.
        :type depth: int, optional
        :param heads: The number of attention heads, defaults to 4.
        :type heads: int, optional
        :param dim_head: The per-head dimension (vanilla transformer only), defaults to 32.
        :type dim_head: int, optional
        :param dropout: Dropout probability, defaults to 0.0.
        :type dropout: float, optional
        :param encoder_type: The sequence backbone: "crossformer" (cross-dimension attention over
            the met covariates, the default) or "transformer" (vanilla encoder), defaults
            to "crossformer".
        :type encoder_type: str, optional
        :param seg_len: The Crossformer segment length, defaults to 3.
        :type seg_len: int, optional
        :param anchored: When True, the network no longer invents forcing. Physical gridded
            precipitation and PET are supplied through ``phys_forcing`` and the network may only
            apply a bounded correction to them, which removes the degenerate optimum where
            effective rainfall collapses to zero and the ODE coasts on storage. Defaults to False
            (legacy generative behaviour).
        :type anchored: bool, optional
        :param use_multiplier: Anchored mode only: whether to learn the bounded multiplier on
            gridded precipitation. Zero-initialized so it starts at exactly 1.0, making epoch 0
            identical to the pure-physics baseline. Defaults to True.
        :type use_multiplier: bool, optional
        :param use_asos_gate: Anchored mode only: whether to add the gated station-innovation
            term ``gate * max(P_station - P_grid, 0)``. A multiplier alone cannot recover a storm
            the grid missed entirely (anything times zero is zero), and stations only ever ADD
            water here -- a dry station does not imply a dry basin, while a wet one is positive
            evidence of rain. The per-basin gate doubles as a learned areal-reduction factor for
            a point observation, so it should fall with station distance. Defaults to False.
        :type use_asos_gate: bool, optional
        """
        super().__init__()
        from flood_forecast.meta_models.merging_model import GatedFusion
        self.seq_len = seq_len
        self.anchored = anchored
        self.use_multiplier = use_multiplier
        self.use_asos_gate = use_asos_gate
        if encoder_type == "crossformer":
            from flood_forecast.transformer_xl.cross_former import CrossformerEncoderOnly
            self.embed = torch.nn.Identity()
            self.encoder = CrossformerEncoderOnly(n_met_features, seq_len, seg_len=seg_len,
                                                  d_model=dim, n_heads=heads, e_layers=depth,
                                                  d_ff=dim * 2, dropout=dropout)
        elif encoder_type == "transformer":
            from flood_forecast.multi_models.crossvivit import Transformer
            self.embed = torch.nn.Linear(n_met_features, dim)
            self.encoder = Transformer(dim, seq_len, depth, heads, dim_head, dim * 2,
                                       dropout=dropout)
        else:
            raise ValueError("encoder_type must be 'crossformer' or 'transformer' but got " +
                             encoder_type)
        self.fusion = GatedFusion(dim, context_dim)
        self.head = torch.nn.Linear(dim, 1 if anchored else 2)
        if anchored:
            # Near-zero (NOT exactly zero) init: the multiplier starts within ~1% of 1.0, so the
            # model begins at the physical baseline, while the head still has a non-zero Jacobian
            # back into the encoder. Exact zeros would leave d(raw)/d(hidden) = 0 and the encoder
            # would receive no gradient at all on the first step -- the same zero-Jacobian trap
            # that GR4ParameterHead hit, and a real risk here because early stopping has already
            # selected epoch-0 checkpoints on this project.
            torch.nn.init.normal_(self.head.weight, std=1e-3)
            torch.nn.init.zeros_(self.head.bias)
            self.register_buffer("log_two", torch.tensor(0.6931471805599453))
            # Static per-basin gate, initialised near 0.18 rather than ~0. A near-zero gate would
            # be nearly untrainable: sigmoid'(-6) is itself ~0.002, so the term's gradient is
            # scaled into irrelevance and short runs with early stopping would never activate it.
            # ~0.18 is also a defensible prior, since the gate acts as an areal-reduction factor
            # for a point observation over a basin.
            self.gate_net = torch.nn.Linear(context_dim, 1)
            torch.nn.init.normal_(self.gate_net.weight, std=1e-3)
            torch.nn.init.constant_(self.gate_net.bias, -1.5)

    def forward(self, met: torch.Tensor, context: torch.Tensor,
                phys_forcing: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates the effective forcing series.

        :param met: Raw meteorology of shape (batch_size, seq_len, n_met_features).
        :type met: torch.Tensor
        :param context: Catchment embeddings of shape (batch_size, context_dim).
        :type context: torch.Tensor
        :param phys_forcing: Required in anchored mode: physical channels of shape
            (batch_size, n_steps, 4) holding [gridded precip mm/hr, gridded PET mm/hr, station
            precip mm/hr, station-observed mask]. Ignored otherwise. Defaults to None.
        :type phys_forcing: torch.Tensor, optional
        :return: Non-negative effective forcing (P_eff, E_eff) of shape (batch_size, n_steps, 2).
        :rtype: torch.Tensor
        """
        n_steps = met.shape[1]
        if n_steps > self.seq_len:
            raise ValueError("Window of %d steps exceeds the encoder's seq_len %d; construct the "
                             "model with seq_len >= the longest window (e.g. the spin-up length)."
                             % (n_steps, self.seq_len))
        if n_steps < self.seq_len:
            # The encoder's positional embedding (and Crossformer's TSA router) are fixed-length,
            # so shorter windows are left-padded by repeating the first step, then the tail sliced.
            padding = met[:, :1, :].expand(-1, self.seq_len - n_steps, -1)
            met = torch.cat([padding, met], dim=1)
        hidden = self.encoder(self.embed(met))[:, -n_steps:, :]
        fused = self.fusion(hidden, context)
        raw = self.head(fused)
        if not self.anchored:
            return torch.nn.functional.softplus(raw)
        if phys_forcing is None:
            raise ValueError("anchored=True requires phys_forcing with [P_grid, PET, P_station, "
                             "station_mask].")
        p_grid = phys_forcing[..., 0].clamp(min=0.0)
        p_eff = p_grid
        if self.use_multiplier:
            # exp(ln2 * tanh(.)) is strictly within [0.5, 2.0] and equals 1.0 at zero input, so
            # precipitation can be corrected but never zeroed out or made absurd.
            p_eff = p_eff * torch.exp(self.log_two * torch.tanh(raw[..., 0]))
        if self.use_asos_gate:
            innovation = (phys_forcing[..., 2].clamp(min=0.0) - p_grid).clamp(min=0.0)
            gate = torch.sigmoid(self.gate_net(context))
            p_eff = p_eff + gate * innovation * phys_forcing[..., 3]
        return torch.stack([p_eff, phys_forcing[..., 1].clamp(min=0.0)], dim=-1)


class HybridGR4Model(torch.nn.Module):
    """
    The end-to-end hybrid: catchment embedding -> GR4 parameters, met window -> effective forcing,
    both pushed through the rigid state-space GR4 ODE to produce streamflow.

    The catchment embedding is an input (produced by a pretrained
    :class:`~flood_forecast.multi_models.catchment_embedding.CatchmentEncoder`), so the same hybrid
    can run with frozen or finetuned context encoders.
    """

    def __init__(self, n_met_features: int, seq_len: int, context_dim: int = 256, dim: int = 64,
                 depth: int = 2, heads: int = 4, n_routing_reservoirs: int = 3,
                 solver_params: Optional[dict] = None, parameter_head_params: Optional[dict] = None,
                 encoder_type: str = "crossformer", snow: bool = False, anchored: bool = False,
                 use_multiplier: bool = True, use_asos_gate: bool = False):
        """
        Initializes the hybrid model.

        :param n_met_features: The number of raw meteorological channels.
        :type n_met_features: int
        :param seq_len: The met window / simulation length in hours.
        :type seq_len: int
        :param context_dim: The catchment embedding dimension, defaults to 256.
        :type context_dim: int, optional
        :param dim: The forcing generator transformer dimension, defaults to 64.
        :type dim: int, optional
        :param depth: The forcing generator depth, defaults to 2.
        :type depth: int, optional
        :param heads: The number of attention heads, defaults to 4.
        :type heads: int, optional
        :param n_routing_reservoirs: Nash cascade size of the GR4 dynamics, defaults to 3.
        :type n_routing_reservoirs: int, optional
        :param solver_params: NeuralODE solver settings, defaults to None which uses fixed-step rk4.
        :type solver_params: dict, optional
        :param parameter_head_params: Keyword arguments for :class:`GR4ParameterHead` (e.g. custom
            parameter ranges), defaults to None.
        :type parameter_head_params: dict, optional
        :param encoder_type: The forcing generator backbone, "crossformer" (default) or
            "transformer".
        :type encoder_type: str, optional
        :param snow: Whether to use the snow-extended dynamics (EXP-HYDRO bucket + GR4). When True,
            :meth:`forward` requires the ``raw_forcing`` argument carrying [T degC, SW] channels and
            the parameter head emits seven parameters. Defaults to False.
        :type snow: bool, optional
        """
        super().__init__()
        from flood_forecast.ode.neural_ode import NeuralODE
        self.snow = snow
        self.forcing_generator = EffectiveForcingGenerator(n_met_features, seq_len,
                                                           context_dim=context_dim, dim=dim,
                                                           depth=depth, heads=heads,
                                                           encoder_type=encoder_type,
                                                           anchored=anchored,
                                                           use_multiplier=use_multiplier,
                                                           use_asos_gate=use_asos_gate)
        if snow:
            self.parameter_head = GR4SnowParameterHead(embedding_dim=context_dim,
                                                       **(parameter_head_params or {}))
            self.dynamics = GR4SnowDynamics(n_routing_reservoirs=n_routing_reservoirs,
                                            learnable=False)
        else:
            self.parameter_head = GR4ParameterHead(embedding_dim=context_dim,
                                                   **(parameter_head_params or {}))
            self.dynamics = GR4Dynamics(n_routing_reservoirs=n_routing_reservoirs, learnable=False)
        if solver_params is None:
            solver_params = {"method": "rk4"}
        self.node = NeuralODE(self.dynamics, **solver_params)
        # times are built per forward from the met window length, so the same model can run
        # spin-up and forecast-horizon windows of different lengths.

    def forward(self, met: torch.Tensor, context: torch.Tensor,
                initial_state: Optional[torch.Tensor] = None,
                raw_forcing: Optional[torch.Tensor] = None,
                initial_snow: Optional[torch.Tensor] = None,
                phys_forcing: Optional[torch.Tensor] = None) -> dict:
        """
        Simulates streamflow for a met window conditioned on catchment embeddings.

        :param met: Raw meteorology of shape (batch_size, seq_len, n_met_features).
        :type met: torch.Tensor
        :param context: Catchment embeddings of shape (batch_size, context_dim).
        :type context: torch.Tensor
        :param initial_state: Initial ODE state of shape (batch_size, state_dim), defaults to None
            which starts the production store at 60% of X1, the routing store at 30% of X3 (moist
            antecedent conditions; a dry production store squares away most effective rain before it
            reaches routing, starving short windows of response) and, with snow, an empty snow store.
        :type initial_state: torch.Tensor, optional
        :param raw_forcing: Required when ``snow=True``: physical channels of shape
            (batch_size, seq_len, 2) holding [temperature degC, shortwave W/m2]. Temperature drives
            the rain/snow partition and melt and is deliberately NOT a learned quantity.
        :type raw_forcing: torch.Tensor, optional
        :param initial_snow: Optional observed SWE in mm of shape (batch_size,) (e.g. SNODAS
            basin means) seeding the snow store on top of ``initial_state`` (or the default
            initialization). Negative entries mean "no observation" and leave the state untouched;
            ignored when ``snow=False``. Defaults to None.
        :type initial_snow: torch.Tensor, optional
        :param phys_forcing: Required when the forcing generator is anchored: physical channels of
            shape (batch_size, seq_len, 4) holding [gridded precip, gridded PET, station precip,
            station mask], all in mm/hr except the mask. Defaults to None.
        :type phys_forcing: torch.Tensor, optional
        :return: A dict with "flow" (batch_size, seq_len), "forcing", "parameters" and "states"
            (for auxiliary supervision such as actual ET and, with snow, SWE via
            ``self.dynamics.swe(states)``).
        :rtype: dict
        """
        parameters = self.parameter_head(context)
        self.dynamics.set_parameters(parameters)
        forcing = self.forcing_generator(met, context, phys_forcing=phys_forcing)
        if self.snow:
            if raw_forcing is None:
                raise ValueError("snow=True requires raw_forcing with [temperature, shortwave].")
            forcing = torch.cat([forcing, raw_forcing], dim=-1)
        times = torch.arange(float(met.shape[1]), device=met.device)
        self.dynamics.set_forcing(forcing, times)
        if initial_state is None:
            initial_state = torch.zeros(met.shape[0], self.dynamics.state_dim, device=met.device)
            offset = 1 if self.snow else 0
            initial_state[:, offset] = 0.6 * parameters[:, 0]
            initial_state[:, offset + 1] = 0.3 * parameters[:, 2]
        if initial_snow is not None and self.snow:
            initial_state = initial_state.clone()
            initial_state[:, 0] = torch.where(initial_snow >= 0.0, initial_snow,
                                              initial_state[:, 0])
        states = self.node(initial_state, times)
        return {"flow": self.dynamics.streamflow(states), "forcing": forcing,
                "parameters": parameters, "states": states}


register_dynamics("gr4", GR4Dynamics)
register_dynamics("gr4_snow", GR4SnowDynamics)
