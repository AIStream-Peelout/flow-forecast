import unittest

import numpy as np
import pandas as pd
import torch

from flood_forecast.ode.physics.forecast_training import (estimate_initial_state,
                                                          match_current_flow,
                                                          train_forecast_model)
from flood_forecast.ode.physics.hydrology import HybridGR4Model
from flood_forecast.preprocessing.catchment_forecast_loader import ForecastWindowDataset


def make_synthetic_frame(n_hours: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Builds a synthetic hourly gauge record with storm-driven flow responses."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n_hours, freq="h", tz="UTC")
    precip = np.zeros(n_hours, dtype=np.float32)
    for start in rng.integers(0, n_hours - 24, size=n_hours // 200):
        precip[start:start + rng.integers(3, 12)] += rng.uniform(1.0, 6.0)
    flow = np.convolve(precip, np.exp(-np.arange(96) / 24.0) / 24.0, mode="full")[:n_hours]
    flow = flow + 0.02  # baseflow
    temp = 285.0 + 10.0 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 365)) + \
        5.0 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    return pd.DataFrame({
        "datetime": index, "precipitation": precip,
        "temperature": temp.astype(np.float32),
        "shortwave_radiation": np.clip(400 * np.sin(2 * np.pi * np.arange(n_hours) / 24), 0,
                                       None).astype(np.float32),
        "pet_mm_hr": np.full(n_hours, 0.05, dtype=np.float32),
        # cfs chosen so the mm/hr conversion with area 100 km2 round-trips the synthetic flow.
        "cfs": (flow * 100.0 / (0.0283168 * 3.6)).astype(np.float32),
    })


MET_COLS = ["precipitation", "temperature", "shortwave_radiation", "pet_mm_hr"]


class TestForecastWindowDataset(unittest.TestCase):
    """Tests for the sliding (spin-up, horizon) window dataset."""

    def setUp(self):
        self.dataset = ForecastWindowDataset(make_synthetic_frame(), MET_COLS, area_km2=100.0,
                                             spinup_hours=240, horizon_hours=96, stride_hours=48)

    def test_window_shapes(self):
        item = self.dataset[0]
        self.assertEqual(item["spinup_met"].shape, (240, 4))
        self.assertEqual(item["horizon_met"].shape, (96, 4))
        self.assertEqual(item["horizon_flow"].shape, (96,))
        self.assertEqual(item["spinup_raw"].shape, (240, 2))
        self.assertGreater(len(self.dataset), 30)

    def test_flow_unit_conversion(self):
        item = self.dataset[0]
        self.assertLess(item["horizon_flow"].max().item(), 5.0)  # mm/hr scale, not cfs
        self.assertGreater(item["horizon_flow"].min().item(), 0.0)

    def test_gappy_windows_are_skipped(self):
        frame = make_synthetic_frame()
        frame.loc[500:1500, "cfs"] = np.nan
        gappy = ForecastWindowDataset(frame, MET_COLS, area_km2=100.0, spinup_hours=240,
                                      horizon_hours=96, stride_hours=48)
        full = self.dataset
        self.assertLess(len(gappy), len(full))
        # Indexed windows may overlap the gap only within the 5% missing-data tolerance.
        for start in gappy.starts:
            overlap = max(0, min(start + 336, 1501) - max(start, 500))
            self.assertLessEqual(overlap, int(0.05 * 336) + 1)

    def test_norm_stats_reusable(self):
        stats = self.dataset.norm_stats
        other = ForecastWindowDataset(make_synthetic_frame(seed=1), MET_COLS, area_km2=100.0,
                                      spinup_hours=240, horizon_hours=96, norm_stats=stats)
        self.assertTrue(np.allclose(other.norm_stats["mean"], stats["mean"]))


class TestStateEstimation(unittest.TestCase):
    """Tests for spin-up state estimation and current-flow assimilation."""

    def make_model(self):
        torch.manual_seed(0)
        return HybridGR4Model(n_met_features=4, seq_len=240, context_dim=16, dim=32, depth=1,
                              snow=True)

    def test_match_current_flow_hits_observation(self):
        model = self.make_model()
        context = torch.randn(3, 16)
        parameters = model.parameter_head(context)
        model.dynamics.set_parameters(parameters)
        state = torch.rand(3, model.dynamics.state_dim) * 50.0
        observed = torch.tensor([0.01, 0.2, 1.5])
        corrected = match_current_flow(model, state, parameters, observed)
        simulated = model.dynamics.streamflow(corrected)
        self.assertTrue(torch.allclose(simulated, observed, rtol=1e-3, atol=1e-4))
        # Snow and production stores must be untouched by assimilation.
        self.assertTrue(torch.equal(corrected[:, :2], state[:, :2]))

    def test_estimate_initial_state_finite_and_detached(self):
        model = self.make_model()
        dataset = ForecastWindowDataset(make_synthetic_frame(), MET_COLS, area_km2=100.0,
                                        spinup_hours=240, horizon_hours=96)
        batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
        state = estimate_initial_state(model, batch, torch.randn(1, 16))
        self.assertEqual(state.shape, (1, model.dynamics.state_dim))
        self.assertTrue(torch.isfinite(state).all())
        self.assertFalse(state.requires_grad)


class TestForecastTraining(unittest.TestCase):
    """Smoke test: a couple of epochs of forecast training should reduce the loss."""

    def test_training_reduces_loss(self):
        torch.manual_seed(0)
        model = HybridGR4Model(n_met_features=4, seq_len=240, context_dim=16, dim=32, depth=1,
                               snow=True,
                               parameter_head_params={"x4_range": (0.5, 48.0)})
        dataset = ForecastWindowDataset(make_synthetic_frame(), MET_COLS, area_km2=100.0,
                                        spinup_hours=240, horizon_hours=96, stride_hours=96)
        losses = train_forecast_model(model, dataset, torch.randn(1, 16), epochs=3, batch_size=8,
                                      lr=5e-3)
        self.assertEqual(len(losses), 3)
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()


class TestForecastReport(unittest.TestCase):
    """Tests for the standard forecast evaluation report."""

    def test_metrics_and_figures(self):
        import tempfile
        import os
        from flood_forecast.ode.physics.forecast_training import forecast_report
        rng = np.random.default_rng(0)
        n, horizon = 12, 336
        timestamps = pd.date_range("2023-01-01", periods=8000, freq="h", tz="UTC")
        t0s = torch.arange(400, 400 + n * 400, 400)
        obs = torch.rand(n, horizon) * 2.0
        sim = obs + 0.1 * torch.randn(n, horizon)
        persist = obs[:, :1].expand(-1, horizon)
        with tempfile.TemporaryDirectory() as out_dir:
            metrics = forecast_report(sim, obs, persist, timestamps, t0s,
                                      full_flow=rng.random(8000).astype(np.float32),
                                      area_km2=100.0, out_dir=out_dir, n_examples=3)
            self.assertIn("day1-3", metrics)
            self.assertIn("skill_vs_persistence_pct", metrics["day1-3"])
            self.assertIn("rmse_cfs", metrics["all"])
            # Near-perfect sim must show large positive skill vs persistence.
            self.assertGreater(metrics["all"]["skill_vs_persistence_pct"], 50.0)
            html = [f for f in os.listdir(out_dir) if f.endswith(".html")]
            self.assertGreaterEqual(len(html), 3)


class TestFFNativeIntegration(unittest.TestCase):
    """End-to-end: the hybrid trains through FF's config-driven PyTorchForecast pipeline."""

    def test_config_driven_training(self):
        import json
        from flood_forecast.time_model import PyTorchForecast
        from flood_forecast.pytorch_training import train_transformer_style
        with open("tests/catchment_hybrid_test.json") as f:
            params = json.load(f)
        model = PyTorchForecast(params["model_name"], params["dataset_params"]["training_path"],
                                params["dataset_params"]["validation_path"],
                                params["dataset_params"]["test_path"], params)
        self.assertGreater(len(model.training), 5)
        src, trg = model.training[0]
        self.assertEqual(src.shape, (168, 5))
        self.assertEqual(trg.shape, (48, 5))
        # Horizon flow must be zeroed in src (no leakage), physical in trg.
        self.assertEqual(src[120:, 0].abs().sum().item(), 0.0)
        self.assertGreater(trg[:, 0].sum().item(), 0.0)
        train_transformer_style(model, params["training_params"], forward_params={})
