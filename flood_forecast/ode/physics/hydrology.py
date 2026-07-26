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


class GR4ParameterHead(torch.nn.Module):
    """
    A hypernetwork head mapping a catchment embedding to the four GR4 parameters.

    Each parameter is squashed by a scaled sigmoid into a physically realistic range, so the ODE can
    never receive degenerate values regardless of the embedding. The output plugs straight into
    :meth:`GR4Dynamics.set_parameters`.
    """

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 64,
                 x1_range: tuple = (10.0, 2000.0), x2_range: tuple = (-10.0, 10.0),
                 x3_range: tuple = (5.0, 500.0), x4_range: tuple = (0.5, 120.0)):
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
        :param x4_range: Bounds (time units) for the unit hydrograph constant, defaults to (0.5, 120).
        :type x4_range: tuple, optional
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim), torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 4),
        )
        # Zero-init the output layer so every catchment starts at mid-range parameters, keeping the
        # scaled sigmoid in its linear region — otherwise early training slams parameters into the
        # bounds where the sigmoid saturates and gradients vanish.
        torch.nn.init.zeros_(self.net[-1].weight)
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
                 encoder_type: str = "crossformer", seg_len: int = 3):
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
        """
        super().__init__()
        from flood_forecast.meta_models.merging_model import GatedFusion
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
        self.head = torch.nn.Linear(dim, 2)

    def forward(self, met: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Generates the effective forcing series.

        :param met: Raw meteorology of shape (batch_size, seq_len, n_met_features).
        :type met: torch.Tensor
        :param context: Catchment embeddings of shape (batch_size, context_dim).
        :type context: torch.Tensor
        :return: Non-negative effective forcing (P_eff, E_eff) of shape (batch_size, seq_len, 2).
        :rtype: torch.Tensor
        """
        hidden = self.encoder(self.embed(met))
        fused = self.fusion(hidden, context)
        return torch.nn.functional.softplus(self.head(fused))


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
                 encoder_type: str = "crossformer"):
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
        """
        super().__init__()
        from flood_forecast.ode.neural_ode import NeuralODE
        self.forcing_generator = EffectiveForcingGenerator(n_met_features, seq_len,
                                                           context_dim=context_dim, dim=dim,
                                                           depth=depth, heads=heads,
                                                           encoder_type=encoder_type)
        self.parameter_head = GR4ParameterHead(embedding_dim=context_dim,
                                               **(parameter_head_params or {}))
        self.dynamics = GR4Dynamics(n_routing_reservoirs=n_routing_reservoirs, learnable=False)
        if solver_params is None:
            solver_params = {"method": "rk4"}
        self.node = NeuralODE(self.dynamics, **solver_params)
        self.register_buffer("times", torch.arange(float(seq_len)))

    def forward(self, met: torch.Tensor, context: torch.Tensor,
                initial_state: Optional[torch.Tensor] = None) -> dict:
        """
        Simulates streamflow for a met window conditioned on catchment embeddings.

        :param met: Raw meteorology of shape (batch_size, seq_len, n_met_features).
        :type met: torch.Tensor
        :param context: Catchment embeddings of shape (batch_size, context_dim).
        :type context: torch.Tensor
        :param initial_state: Initial ODE state of shape (batch_size, state_dim), defaults to None
            which starts the production store at 60% of X1 and the routing store at 30% of X3
            (moist antecedent conditions; a dry production store squares away most effective rain
            before it reaches routing, starving short windows of response).
        :type initial_state: torch.Tensor, optional
        :return: A dict with "flow" (batch_size, seq_len), "forcing", "parameters" and "states"
            (for auxiliary supervision such as actual ET).
        :rtype: dict
        """
        parameters = self.parameter_head(context)
        self.dynamics.set_parameters(parameters)
        forcing = self.forcing_generator(met, context)
        self.dynamics.set_forcing(forcing, self.times)
        if initial_state is None:
            initial_state = torch.zeros(met.shape[0], self.dynamics.state_dim, device=met.device)
            initial_state[:, 0] = 0.6 * parameters[:, 0]
            initial_state[:, 1] = 0.3 * parameters[:, 2]
        states = self.node(initial_state, self.times)
        return {"flow": self.dynamics.streamflow(states), "forcing": forcing,
                "parameters": parameters, "states": states}


register_dynamics("gr4", GR4Dynamics)
