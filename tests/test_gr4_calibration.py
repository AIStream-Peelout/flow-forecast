"""Tests for classical GR4-snow calibration (the warm-start stage-one)."""
import unittest

import torch

from flood_forecast.ode.physics.gr4_calibration import (
    DEFAULT_BOUNDS,
    PARAMETER_NAMES,
    calibrate_gr4_snow,
    nse,
    simulate_gr4_snow,
    unit_to_parameters,
)


def _synthetic_forcing(n_hours: int, seed: int = 7) -> torch.Tensor:
    """
    Builds a warm-season synthetic forcing series with clustered storm pulses.

    :param n_hours: Number of hourly steps.
    :type n_hours: int
    :param seed: Random seed controlling storm placement, defaults to 7.
    :type seed: int
    :return: Forcing of shape (n_hours, 4): [P mm/hr, PET mm/hr, T degC, SW].
    :rtype: torch.Tensor
    """
    generator = torch.Generator().manual_seed(seed)
    precip = torch.zeros(n_hours)
    storm_starts = torch.randint(0, n_hours - 12, (n_hours // 120,), generator=generator)
    for start in storm_starts:
        length = int(torch.randint(3, 12, (1,), generator=generator))
        precip[start:start + length] += torch.rand(length, generator=generator) * 6.0
    hours = torch.arange(n_hours, dtype=torch.float32)
    pet = 0.12 * (1.0 + torch.sin(2 * torch.pi * (hours % 24) / 24.0 - 1.5)).clamp(min=0.0)
    temperature = torch.full((n_hours,), 15.0)
    shortwave = torch.zeros(n_hours)
    return torch.stack([precip, pet, temperature, shortwave], dim=-1)


class TestUnitToParameters(unittest.TestCase):
    """Bounds and ordering of the unit-cube parameter mapping."""

    def test_bounds_and_ordering(self):
        unit = torch.rand(64, 7)
        params = unit_to_parameters(unit)
        for column, name in enumerate(PARAMETER_NAMES[:6]):
            lower, upper = DEFAULT_BOUNDS[name]
            self.assertTrue(bool((params[:, column] >= lower - 1e-5).all()), name)
            self.assertTrue(bool((params[:, column] <= upper + 1e-5).all()), name)
        self.assertTrue(bool((params[:, 6] <= params[:, 5] + 1e-6).all()),
                        "Tmin must never exceed Tmax")

    def test_log_uniform_median_is_below_midpoint(self):
        unit = torch.full((1, 7), 0.5)
        params = unit_to_parameters(unit)
        self.assertLess(float(params[0, 0]), 300.0)  # X1 log-median ~ 141 mm, not 1005
        self.assertLess(float(params[0, 3]), 20.0)  # X4 log-median ~ 15.5 h, not 61


class TestCalibration(unittest.TestCase):
    """Round-trip parameter recovery through the batched simulator."""

    def test_recovers_synthetic_basin(self):
        forcing = _synthetic_forcing(2880)
        true_params = torch.tensor([[220.0, -0.5, 60.0, 8.0, 0.1, 0.5, -0.5]])
        observed = simulate_gr4_snow(true_params, forcing)[0]
        self.assertGreater(float(observed[720:].std()), 1e-4,
                           "synthetic basin must actually respond to storms")
        result = calibrate_gr4_snow(forcing, observed, n_random=96, chunk_size=48,
                                    refine_rounds=2, refine_top=8, refine_samples=64,
                                    warmup_hours=720, substeps=2, seed=3)
        self.assertGreater(result["nse"], 0.8)
        self.assertGreater(result["nse"], result["midpoint_nse"])
        self.assertEqual(set(result["parameters"]), set(PARAMETER_NAMES))

    def test_nse_masks_invalid_hours(self):
        simulated = torch.zeros(2, 100)
        observed = torch.ones(100)
        observed[50:] = float("nan")
        scores = nse(simulated, observed)
        self.assertTrue(bool(torch.isfinite(scores).all()))


if __name__ == "__main__":
    unittest.main()
