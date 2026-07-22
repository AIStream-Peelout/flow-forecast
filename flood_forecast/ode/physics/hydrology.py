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
        forcing = self.forcing_at(t)
        precip, pet = forcing[:, 0].clamp(min=0.0), forcing[:, 1].clamp(min=0.0)
        params = self.gr4_parameters()
        x1, x2, x3, x4 = params.unbind(-1)
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


register_dynamics("gr4", GR4Dynamics)
