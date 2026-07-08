"""
Reusable Neural ODE building blocks and an end-to-end forecasting model.

``NeuralODE`` integrates any :class:`~flood_forecast.ode.dynamics.BaseDynamics` with torchdiffeq and can be
embedded in other models the same way an ``nn.GRU`` would be. ``ODEForecast`` is the trainable model exposed
to the JSON config system through ``pytorch_model_dict``.
"""
from typing import Dict, Optional

import torch
from torchdiffeq import odeint, odeint_adjoint

from flood_forecast.ode.dynamics import BaseDynamics, build_dynamics


class NeuralODE(torch.nn.Module):
    """
    A generic wrapper that integrates a dynamics module over a set of time points with torchdiffeq.
    """

    def __init__(self, dynamics: BaseDynamics, method: str = "dopri5", rtol: float = 1e-4, atol: float = 1e-5,
                 adjoint: bool = False, solver_options: Optional[Dict] = None):
        """
        Initializes the NeuralODE wrapper.

        :param dynamics: The right-hand side module defining dstate/dt = f(t, state).
        :type dynamics: BaseDynamics
        :param method: The torchdiffeq solver to use (e.g. "dopri5", "rk4", "implicit_adams"),
            defaults to "dopri5".
        :type method: str, optional
        :param rtol: Relative tolerance for adaptive-step solvers, defaults to 1e-4.
        :type rtol: float, optional
        :param atol: Absolute tolerance for adaptive-step solvers, defaults to 1e-5.
        :type atol: float, optional
        :param adjoint: Whether to use the memory-efficient adjoint method for the backward pass,
            defaults to False.
        :type adjoint: bool, optional
        :param solver_options: Extra options forwarded to the solver (e.g. {"step_size": 0.5} for fixed-step
            methods), defaults to None.
        :type solver_options: Dict, optional
        """
        super().__init__()
        self.dynamics = dynamics
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.solver_options = solver_options

    def forward(self, initial_state: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Integrates the dynamics from the initial state over the given time points.

        :param initial_state: The state at times[0], of shape (batch_size, state_dim).
        :type initial_state: torch.Tensor
        :param times: A 1D increasing tensor of time points at which to evaluate the solution.
        :type times: torch.Tensor
        :return: The solution states of shape (batch_size, len(times), state_dim).
        :rtype: torch.Tensor
        """
        integrator = odeint_adjoint if self.adjoint else odeint
        states = integrator(self.dynamics, initial_state, times, method=self.method, rtol=self.rtol,
                            atol=self.atol, options=self.solver_options)
        return states.permute(1, 0, 2)


class ODEForecast(torch.nn.Module):
    """
    An encoder-ODE-decoder model for time series forecasting.

    A GRU encoder maps the history window to the initial ODE state, the state is integrated across the
    forecast horizon by a :class:`NeuralODE`, and a linear decoder maps the states to the target values.
    The dynamics (learned, physics-based or hybrid) are selected via the dynamics_params dict in the
    JSON config.
    """

    def __init__(self, n_time_series: int, n_target: int, forecast_length: int, dynamics_params: Dict,
                 solver_params: Optional[Dict] = None, encoder_hidden_dim: int = 32, encoder_layers: int = 1,
                 time_step: float = 1.0):
        """
        Initializes the ODEForecast model.

        :param n_time_series: The number of input time series features.
        :type n_time_series: int
        :param n_target: The number of output targets.
        :type n_target: int
        :param forecast_length: The number of future time steps to forecast.
        :type forecast_length: int
        :param dynamics_params: The dynamics configuration with a "type" key selecting a class from
            ode_dynamics_dict; remaining keys are passed to its constructor.
        :type dynamics_params: Dict
        :param solver_params: Keyword arguments for the NeuralODE wrapper (method, rtol, atol, adjoint,
            solver_options), defaults to None.
        :type solver_params: Dict, optional
        :param encoder_hidden_dim: The hidden size of the GRU encoder, defaults to 32.
        :type encoder_hidden_dim: int, optional
        :param encoder_layers: The number of GRU encoder layers, defaults to 1.
        :type encoder_layers: int, optional
        :param time_step: The physical time between consecutive forecast steps, used to build the
            integration grid, defaults to 1.0.
        :type time_step: float, optional
        """
        super().__init__()
        dynamics = build_dynamics(dynamics_params)
        if solver_params is None:
            solver_params = {}
        self.node = NeuralODE(dynamics, **solver_params)
        self.encoder = torch.nn.GRU(n_time_series, encoder_hidden_dim, encoder_layers, batch_first=True)
        self.state_projection = torch.nn.Linear(encoder_hidden_dim, dynamics.state_dim)
        self.decoder = torch.nn.Linear(dynamics.state_dim, n_target)
        times = torch.linspace(0.0, forecast_length * time_step, forecast_length + 1)
        self.register_buffer("times", times)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ODEForecast model.

        :param x: Input tensor of shape (batch_size, sequence_length, n_time_series).
        :type x: torch.Tensor
        :return: Forecasts of shape (batch_size, forecast_length) if n_target is 1, otherwise of shape
            (batch_size, forecast_length, n_target).
        :rtype: torch.Tensor
        """
        _, hidden = self.encoder(x)
        initial_state = self.state_projection(hidden[-1])
        states = self.node(initial_state, self.times)[:, 1:, :]
        out = self.decoder(states)
        if self.decoder.out_features == 1:
            return out[:, :, 0]
        return out
