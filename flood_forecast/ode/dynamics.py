"""
Right-hand side (dynamics) definitions for neural and physics-based ODEs.

Each dynamics class defines ``dstate/dt = f(t, state)`` and exposes a ``state_dim`` attribute so downstream
modules know the size of the state vector. The intent is that the *structure* of an equation comes from
domain science (rainfall-runoff, epidemiology, glaciology, etc.) while its parameters are ``torch.nn.Parameter``
objects learned from data. New dynamics are made available to JSON configs by adding them to
``ode_dynamics_dict`` (or calling :func:`register_dynamics`) and selecting them with the ``"type"`` key of
``dynamics_params``.
"""
import math
from typing import Dict, List, Optional

import torch

activation_dict = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "softplus": torch.nn.Softplus,
    "sigmoid": torch.nn.Sigmoid,
    "elu": torch.nn.ELU,
    "gelu": torch.nn.GELU,
}


def _inverse_softplus(value: float) -> float:
    """
    Computes the inverse of the softplus function.

    Used to initialize raw (unconstrained) parameters so that ``softplus(raw) == value`` at the start of
    training, keeping physical rate constants positive throughout optimization.

    :param value: The desired positive value after the softplus transform.
    :type value: float
    :return: The pre-softplus (raw) value.
    :rtype: float
    """
    return math.log(math.expm1(value))


class BaseDynamics(torch.nn.Module):
    """
    Abstract base class for ODE right-hand sides.

    Subclasses must set ``self.state_dim`` in their constructor and implement ``forward(t, state)`` returning
    ``dstate/dt`` with the same shape as ``state``. The signature matches what ``torchdiffeq.odeint`` expects.
    """

    state_dim: int

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the time derivative of the state.

        :param t: A scalar tensor with the current integration time.
        :type t: torch.Tensor
        :param state: The current state of shape (batch_size, state_dim).
        :type state: torch.Tensor
        :return: The derivative dstate/dt of shape (batch_size, state_dim).
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class MLPDynamics(BaseDynamics):
    """
    Fully learned dynamics where ``f(t, state)`` is a multi-layer perceptron (the classic Neural ODE).
    """

    def __init__(self, state_dim: int, hidden_layers: Optional[List[int]] = None, activation: str = "tanh",
                 time_dependent: bool = False):
        """
        Builds the MLP from a layer specification.

        :param state_dim: The dimensionality of the ODE state vector.
        :type state_dim: int
        :param hidden_layers: Sizes of the hidden layers, defaults to [64, 64].
        :type hidden_layers: List[int], optional
        :param activation: Name of the activation function, must be a key in activation_dict, defaults to "tanh".
        :type activation: str, optional
        :param time_dependent: Whether to concatenate the integration time t to the MLP input, defaults to False.
        :type time_dependent: bool, optional
        """
        super().__init__()
        if activation not in activation_dict:
            raise KeyError("Activation " + activation + " not found in activation_dict. Please add it.")
        self.state_dim = state_dim
        self.time_dependent = time_dependent
        if hidden_layers is None:
            hidden_layers = [64, 64]
        in_dim = state_dim + 1 if time_dependent else state_dim
        layers: List[torch.nn.Module] = []
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(activation_dict[activation]())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, state_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the learned time derivative of the state.

        :param t: A scalar tensor with the current integration time.
        :type t: torch.Tensor
        :param state: The current state of shape (batch_size, state_dim).
        :type state: torch.Tensor
        :return: The derivative dstate/dt of shape (batch_size, state_dim).
        :rtype: torch.Tensor
        """
        if self.time_dependent:
            t_column = t * torch.ones(state.shape[0], 1, dtype=state.dtype, device=state.device)
            state = torch.cat([state, t_column], dim=-1)
        return self.net(state)


class LinearReservoir(BaseDynamics):
    """
    Linear reservoir dynamics for rainfall-runoff modeling.

    A single reservoir follows the classic recession equation ``dS/dt = -k * S``. With ``n_reservoirs > 1``
    the reservoirs are chained into a Nash cascade where the outflow of reservoir i feeds reservoir i + 1:
    ``dS_i/dt = k_{i-1} * S_{i-1} - k_i * S_i``. The rate constants k are learnable and kept positive via a
    softplus transform.
    """

    def __init__(self, n_reservoirs: int = 1, k_init: float = 0.1, learnable: bool = True):
        """
        Initializes the reservoir cascade.

        :param n_reservoirs: The number of reservoirs in the cascade, defaults to 1.
        :type n_reservoirs: int, optional
        :param k_init: The initial value of each rate constant k, defaults to 0.1.
        :type k_init: float, optional
        :param learnable: Whether the rate constants are trained by gradient descent, defaults to True.
        :type learnable: bool, optional
        """
        super().__init__()
        self.state_dim = n_reservoirs
        raw_k = torch.full((n_reservoirs,), _inverse_softplus(k_init))
        self.raw_k = torch.nn.Parameter(raw_k, requires_grad=learnable)

    @property
    def k(self) -> torch.Tensor:
        """
        The positive rate constants of the reservoirs.

        :return: A tensor of shape (n_reservoirs,) with the softplus-transformed rate constants.
        :rtype: torch.Tensor
        """
        return torch.nn.functional.softplus(self.raw_k)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the storage derivatives of the reservoir cascade.

        :param t: A scalar tensor with the current integration time (unused, the system is autonomous).
        :type t: torch.Tensor
        :param state: The current storages of shape (batch_size, n_reservoirs).
        :type state: torch.Tensor
        :return: The derivative dS/dt of shape (batch_size, n_reservoirs).
        :rtype: torch.Tensor
        """
        outflow = self.k * state
        inflow = torch.zeros_like(outflow)
        inflow[..., 1:] = outflow[..., :-1]
        return inflow - outflow


class SEIRDynamics(BaseDynamics):
    """
    SEIR compartmental dynamics for epidemiological forecasting.

    The state is the population fractions (S, E, I, R) governed by ``dS/dt = -beta * S * I``,
    ``dE/dt = beta * S * I - sigma * E``, ``dI/dt = sigma * E - gamma * I`` and ``dR/dt = gamma * I``.
    The transmission (beta), incubation (sigma) and recovery (gamma) rates are learnable and kept positive
    via a softplus transform.
    """

    def __init__(self, beta_init: float = 0.3, sigma_init: float = 0.2, gamma_init: float = 0.1,
                 learnable: bool = True):
        """
        Initializes the SEIR rate parameters.

        :param beta_init: Initial transmission rate, defaults to 0.3.
        :type beta_init: float, optional
        :param sigma_init: Initial incubation rate (1 / latent period), defaults to 0.2.
        :type sigma_init: float, optional
        :param gamma_init: Initial recovery rate (1 / infectious period), defaults to 0.1.
        :type gamma_init: float, optional
        :param learnable: Whether the rates are trained by gradient descent, defaults to True.
        :type learnable: bool, optional
        """
        super().__init__()
        self.state_dim = 4
        raw = torch.tensor([_inverse_softplus(beta_init), _inverse_softplus(sigma_init),
                            _inverse_softplus(gamma_init)])
        self.raw_rates = torch.nn.Parameter(raw, requires_grad=learnable)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the SEIR compartment derivatives.

        :param t: A scalar tensor with the current integration time (unused, the system is autonomous).
        :type t: torch.Tensor
        :param state: The current compartments (S, E, I, R) of shape (batch_size, 4).
        :type state: torch.Tensor
        :return: The derivatives of shape (batch_size, 4).
        :rtype: torch.Tensor
        """
        beta, sigma, gamma = torch.nn.functional.softplus(self.raw_rates).unbind(-1)
        s, e, i, _ = state.unbind(-1)
        new_infections = beta * s * i
        ds = -new_infections
        de = new_infections - sigma * e
        di = sigma * e - gamma * i
        dr = gamma * i
        return torch.stack([ds, de, di, dr], dim=-1)


class HybridDynamics(BaseDynamics):
    """
    A physics-based dynamics with an additive learned residual: ``f = f_physics + f_mlp``.

    This is the universal differential equation pattern: the equation structure comes from domain science
    while the MLP absorbs unmodeled parameter interactions and forcings.
    """

    def __init__(self, physics: Dict, residual: Optional[Dict] = None):
        """
        Builds the physics term and the residual MLP.

        :param physics: A dynamics_params dict (with a "type" key) describing the physics-based term.
        :type physics: Dict
        :param residual: Keyword arguments for the residual MLPDynamics (e.g. hidden_layers, activation),
            defaults to None which uses the MLPDynamics defaults.
        :type residual: Dict, optional
        """
        super().__init__()
        self.physics = build_dynamics(physics)
        self.state_dim = self.physics.state_dim
        if residual is None:
            residual = {}
        self.residual = MLPDynamics(self.state_dim, **residual)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the physics derivative plus the learned residual.

        :param t: A scalar tensor with the current integration time.
        :type t: torch.Tensor
        :param state: The current state of shape (batch_size, state_dim).
        :type state: torch.Tensor
        :return: The combined derivative of shape (batch_size, state_dim).
        :rtype: torch.Tensor
        """
        return self.physics(t, state) + self.residual(t, state)


ode_dynamics_dict = {
    "mlp": MLPDynamics,
    "linear_reservoir": LinearReservoir,
    "seir": SEIRDynamics,
    "hybrid": HybridDynamics,
}


def register_dynamics(name: str, dynamics_class: type) -> None:
    """
    Registers a new dynamics class so it can be referenced by name in JSON configs.

    :param name: The name to use as the "type" value in dynamics_params.
    :type name: str
    :param dynamics_class: A subclass of BaseDynamics.
    :type dynamics_class: type
    :return: None
    :rtype: None
    """
    ode_dynamics_dict[name] = dynamics_class


def build_dynamics(dynamics_params: Dict) -> BaseDynamics:
    """
    Instantiates a dynamics class from a JSON-style parameter dict.

    The "type" key selects the class from ode_dynamics_dict and all remaining keys are passed to its
    constructor as keyword arguments.

    :param dynamics_params: The dynamics configuration, e.g. {"type": "seir", "beta_init": 0.4}.
    :type dynamics_params: Dict
    :return: The instantiated dynamics module.
    :rtype: BaseDynamics
    """
    params = dict(dynamics_params)
    dynamics_type = params.pop("type")
    if dynamics_type not in ode_dynamics_dict:
        raise KeyError("Error the dynamics " + dynamics_type +
                       " was not found in ode_dynamics_dict. Please add it.")
    return ode_dynamics_dict[dynamics_type](**params)
