"""
Domain-specific physics-based ODE dynamics.

Each submodule covers one scientific domain (e.g. :mod:`~flood_forecast.ode.physics.hydrology`) and
self-registers its dynamics classes in ``ode_dynamics_dict`` at import time so they can be selected from
JSON configs. Generic, domain-agnostic building blocks belong in :mod:`flood_forecast.ode.dynamics` instead.
"""
from flood_forecast.ode.physics.hydrology import GR4Dynamics

__all__ = ["GR4Dynamics"]
