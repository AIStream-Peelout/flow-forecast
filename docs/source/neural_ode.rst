Neural ODE Models
=================

The ``flood_forecast.ode`` package provides Neural Ordinary Differential Equation models for time series
forecasting. It supports three modes of operation:

1. **Fully learned dynamics** — the ODE right-hand side is a neural network trained end-to-end
   (the classic Neural ODE of Chen et al., 2018).
2. **Physics-based dynamics** — the equation structure comes from domain science (e.g. rainfall-runoff,
   epidemiology) and only its physical constants (rate parameters, etc.) are learned from data.
3. **Hybrid dynamics** — a physics equation plus an additive learned residual, for cases where the known
   equation is correct but incomplete (the universal differential equation pattern).

In all three cases the ODE is *selected and configured entirely from the JSON config*, while the equations
themselves live in Python classes registered in ``ode_dynamics_dict``.

Architecture
------------

The package is split into three composable pieces:

* **Dynamics** (:mod:`flood_forecast.ode.dynamics`) — classes defining ``dstate/dt = f(t, state)``.
  Each subclasses :class:`~flood_forecast.ode.dynamics.BaseDynamics`, exposes a ``state_dim`` attribute,
  and is registered by name in ``ode_dynamics_dict``.
* **NeuralODE** (:class:`~flood_forecast.ode.neural_ode.NeuralODE`) — a reusable wrapper that integrates
  any dynamics module with `torchdiffeq <https://github.com/rtqichen/torchdiffeq>`_. It can be embedded
  inside other models the same way an ``nn.GRU`` would be.
* **ODEForecast** (:class:`~flood_forecast.ode.neural_ode.ODEForecast`) — the end-to-end trainable model
  registered as ``"NeuralODE"`` in ``pytorch_model_dict``. A GRU encoder maps the history window to the
  initial ODE state, the state is integrated across the forecast horizon, and a linear decoder maps the
  integrated states to the target values.

Training a Neural ODE model
---------------------------

Neural ODE models are trained through the standard config-driven flow. Set ``"model_name": "NeuralODE"``
and describe the ODE inside ``model_params``:

.. code-block:: bash

    python flood_forecast/trainer.py -p tests/neural_ode_test.json

or from Python:

.. code-block:: python

    import json
    from flood_forecast.trainer import train_function

    with open("tests/neural_ode_test.json") as f:
        params = json.load(f)
    trained_model = train_function("PyTorch", params)

Model parameters
----------------

The ``model_params`` section accepts the following keys:

.. code-block:: javascript

    "model_params": {
        "n_time_series": 3,          // number of input feature columns
        "n_target": 1,               // number of target columns
        "forecast_length": 1,        // steps to forecast per forward pass
        "encoder_hidden_dim": 32,    // GRU encoder hidden size (default 32)
        "encoder_layers": 1,         // GRU encoder layers (default 1)
        "time_step": 1.0,            // physical time between forecast steps (default 1.0)
        "dynamics_params": { ... },  // which ODE to use, see below
        "solver_params": { ... }     // torchdiffeq solver settings, see below
    }

``dynamics_params`` selects the ODE via its ``"type"`` key; every other key is passed to the dynamics
class constructor. The built-in types are:

``"mlp"`` — fully learned dynamics
    .. code-block:: javascript

        "dynamics_params": {
            "type": "mlp",
            "state_dim": 16,             // dimensionality of the latent ODE state
            "hidden_layers": [64, 64],   // MLP layer sizes (default [64, 64])
            "activation": "tanh",        // tanh, relu, softplus, sigmoid, elu, or gelu
            "time_dependent": false      // concatenate t to the MLP input
        }

``"linear_reservoir"`` — rainfall-runoff recession / Nash cascade
    A single reservoir follows ``dS/dt = -k * S``; with ``n_reservoirs > 1`` the outflow of each
    reservoir feeds the next (a Nash cascade). The rate constants ``k`` are learnable parameters kept
    positive through a softplus transform.

    .. code-block:: javascript

        "dynamics_params": {
            "type": "linear_reservoir",
            "n_reservoirs": 8,
            "k_init": 0.1,       // initial rate constant value
            "learnable": true    // set false to freeze k at k_init
        }

``"seir"`` — SEIR compartmental epidemiological model
    The state is the population fractions (S, E, I, R). The transmission rate beta, incubation rate
    sigma, and recovery rate gamma are learnable and kept positive.

    .. code-block:: javascript

        "dynamics_params": {
            "type": "seir",
            "beta_init": 0.3,
            "sigma_init": 0.2,
            "gamma_init": 0.1,
            "learnable": true
        }

``"hybrid"`` — physics plus a learned residual
    Computes ``f = f_physics + f_mlp``. Use this when the scientific equation captures the dominant
    dynamics but unmodeled interactions or forcings remain.

    .. code-block:: javascript

        "dynamics_params": {
            "type": "hybrid",
            "physics": {"type": "linear_reservoir", "n_reservoirs": 8, "k_init": 0.1},
            "residual": {"hidden_layers": [32], "activation": "tanh"}
        }

``solver_params`` configures the torchdiffeq integrator:

.. code-block:: javascript

    "solver_params": {
        "method": "dopri5",   // any torchdiffeq solver: dopri5, rk4, euler, implicit_adams, ...
        "rtol": 1e-4,         // relative tolerance (adaptive-step solvers only)
        "atol": 1e-5,         // absolute tolerance (adaptive-step solvers only)
        "adjoint": false,     // memory-efficient adjoint backward pass
        "solver_options": {}  // extra solver options, e.g. {"step_size": 0.5} for fixed-step methods
    }

Practical guidance: fixed-step ``rk4`` is fast and usually sufficient for smooth dynamics and short
horizons; adaptive ``dopri5`` (the default) is more accurate for stiff or fast-changing dynamics;
``adjoint: true`` reduces memory at the cost of extra compute and matters mainly for long integration
horizons or large states.

Adding your own physical ODE
----------------------------

The intended workflow for new domains (glacier calving, water quality, etc.) is to write the equation once
in Python with the unknown constants as learnable parameters, then drive all experimentation from JSON.

1. Subclass :class:`~flood_forecast.ode.dynamics.BaseDynamics`, set ``self.state_dim``, and implement
   ``forward(t, state)`` returning ``dstate/dt``:

.. code-block:: python

    import torch
    from flood_forecast.ode.dynamics import BaseDynamics, _inverse_softplus, register_dynamics


    class GlacierCalving(BaseDynamics):
        """dV/dt = -c * V**(2/3), a toy calving law with learnable coefficient c."""

        def __init__(self, c_init: float = 0.05, learnable: bool = True):
            super().__init__()
            self.state_dim = 1
            raw_c = torch.full((1,), _inverse_softplus(c_init))
            self.raw_c = torch.nn.Parameter(raw_c, requires_grad=learnable)

        def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            c = torch.nn.functional.softplus(self.raw_c)
            return -c * state.clamp(min=0) ** (2.0 / 3.0)

2. Register it:

.. code-block:: python

    register_dynamics("glacier_calving", GlacierCalving)

3. Reference it from any config:

.. code-block:: javascript

    "dynamics_params": {"type": "glacier_calving", "c_init": 0.05}

The class composes with everything automatically — it can be wrapped in ``"hybrid"`` to add a learned
residual, integrated with any solver, and its parameters receive gradients through the ODE solve.

Conventions for physical parameters:

* Keep rate constants positive with ``softplus(raw_parameter)``; initialize the raw parameter with
  ``_inverse_softplus(desired_value)`` so training starts at a physically sensible value.
* Expose a ``learnable`` flag so a parameter can be frozen at a literature value from the config.
* Autonomous systems ignore ``t``; keep it in the signature since torchdiffeq always passes it.

Current limitations
-------------------

* **No external forcing during integration.** The dynamics are autonomous: known future covariates
  (e.g. precipitation ``P(t)``) are not injected into the ODE mid-integration. Their influence enters
  through the encoder's initial state and, in hybrid mode, the residual term.
* **ODEs only.** Delay differential equations (DDEs) are not supported by torchdiffeq; incubation-style
  delays should be approximated with additional compartments (as SEIR does with the E compartment).
* **Regular time grid.** Integration runs on an evenly spaced grid derived from ``forecast_length`` and
  ``time_step``. Irregularly sampled data is not yet supported.

API reference
-------------

.. automodule:: flood_forecast.ode.dynamics
    :members:

.. automodule:: flood_forecast.ode.neural_ode
    :members:
