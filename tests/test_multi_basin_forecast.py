import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from flood_forecast.ode.physics.forecast_training import HybridGR4MultiBasin
from flood_forecast.preprocessing.pytorch_loaders import MultiBasinWindowLoader
from flood_forecast.pytorch_training import train_transformer_style
from flood_forecast.time_model import PyTorchForecast

RELEVANT_COLS = ["cfs", "precipitation", "temperature", "pet_mm_hr", "p01m", "temp_lapse_k",
                 "sw_raw"]
SWE_RELEVANT_COLS = RELEVANT_COLS + ["snodas_swe_mm"]
SCALED_COLS = ["precipitation", "temperature", "pet_mm_hr", "p01m"]
TRAIN_END = "2022-10-01"
VALID_END = "2022-12-15"
SWE_TEST_VALUE = 150.0
SWE_COVERAGE_END = "2022-04-30"


def make_gauge_frame(n_hours: int, seed: int, response_hours: float) -> pd.DataFrame:
    """
    Builds one synthetic hourly gauge record with storm-driven flow and a gappy p01m column.

    :param n_hours: The record length in hours.
    :type n_hours: int
    :param seed: The random seed (basin identity).
    :type seed: int
    :param response_hours: The recession time scale of the synthetic hydrograph.
    :type response_hours: float
    :return: A frame with datetime, cfs and met columns.
    :rtype: pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n_hours, freq="h", tz="UTC")
    precip = np.zeros(n_hours, dtype=np.float32)
    for start in rng.integers(0, n_hours - 24, size=n_hours // 150):
        precip[start:start + rng.integers(3, 12)] += rng.uniform(1.0, 6.0)
    kernel = np.exp(-np.arange(int(response_hours * 4)) / response_hours) / response_hours
    flow = np.convolve(precip, kernel, mode="full")[:n_hours] + 0.02
    hours = np.arange(n_hours)
    p01m = precip * rng.uniform(0.7, 1.3, size=n_hours).astype(np.float32)
    p01m[: n_hours // 3] = np.nan  # ASOS-like late start; must be filled from NLDAS precip
    return pd.DataFrame({
        "datetime": index,
        "cfs": (flow * 100.0 / (0.0283168 * 3.6)).astype(np.float32),
        "precipitation": precip,
        "temperature": (285.0 + 10.0 * np.sin(2 * np.pi * hours / (24 * 365)) +
                        5.0 * np.sin(2 * np.pi * hours / 24)).astype(np.float32),
        "shortwave_radiation": np.clip(400 * np.sin(2 * np.pi * hours / 24), 0,
                                       None).astype(np.float32),
        "pet_mm_hr": np.full(n_hours, 0.05, dtype=np.float32),
        "p01m": p01m,
    })


def build_manifest(directory: str, with_swe: bool = False) -> str:
    """
    Writes three synthetic gauge CSVs, an embedding bank covering two of them, and the manifest.

    :param directory: The directory to write the fixture files into.
    :type directory: str
    :param with_swe: Whether to also write daily SWE series (for basinA and basinB only, with
        coverage ending SWE_COVERAGE_END so later windows exercise the sentinel), defaults to
        False.
    :type with_swe: bool, optional
    :return: The manifest JSON path.
    :rtype: str
    """
    sites = [("basinA", 1, 24.0, "train", True), ("basinB", 2, 48.0, "train", False),
             ("basinC", 3, 36.0, "holdout", True)]
    basins = []
    for site, seed, response, split, has_embedding in sites:
        frame = make_gauge_frame(10000, seed, response)
        csv_path = os.path.join(directory, site + ".csv")
        frame.to_csv(csv_path, index=False)
        train_rows = frame[frame["datetime"] < pd.Timestamp(TRAIN_END, tz="UTC")]
        flow_mm = train_rows["cfs"] * 0.0283168 * 3.6 / 100.0
        met_stats = {col: [float(train_rows[col].mean()), float(train_rows[col].std())]
                     for col in SCALED_COLS if col in frame.columns}
        met_stats["p01m"] = met_stats["precipitation"]
        entry = {"site_id": site, "csv_path": csv_path, "area_sq_km": 100.0,
                 "temp_offset_c": -1.5, "flow_scale_mm_hr": float(flow_mm.std()),
                 "met_stats": met_stats, "has_embedding": has_embedding, "split": split}
        if with_swe and site != "basinC":
            days = pd.date_range("2022-01-01", SWE_COVERAGE_END, freq="D")
            swe = pd.DataFrame({"datetime": days.strftime("%Y-%m-%d"),
                                "snodas_swe_mm": SWE_TEST_VALUE})
            entry["swe_csv_path"] = os.path.join(directory, site + "_snodas_swe.csv")
            swe.to_csv(entry["swe_csv_path"], index=False)
        basins.append(entry)
    torch.save({"site_ids": ["basinA", "basinC"], "embeddings": torch.randn(2, 8)},
               os.path.join(directory, "embeddings.pt"))
    manifest = {
        "embedding_path": os.path.join(directory, "embeddings.pt"),
        "preprocessing": {"fill_from": {"p01m": "precipitation"},
                          "copy_cols": {"sw_raw": "shortwave_radiation",
                                        "precip_raw": "precipitation",
                                        "pet_raw": "pet_mm_hr",
                                        "asos_raw": "p01m"},
                          "observed_mask_cols": {"asos_observed": "p01m"},
                          "lapse": {"source": "temperature", "target": "temp_lapse_k"},
                          "swe_col": "snodas_swe_mm"},
        "basins": basins,
    }
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def make_params(manifest_path: str) -> dict:
    """
    Builds the FF config dict for the synthetic multi-basin run.

    :param manifest_path: The manifest JSON path.
    :type manifest_path: str
    :return: The config dict.
    :rtype: dict
    """
    return {
        "model_name": "HybridGR4MultiBasin",
        "model_type": "PyTorch",
        "model_params": {"n_time_series": len(RELEVANT_COLS) + 1, "spinup_length": 120,
                         "forecast_length": 48, "raw_temp_index": 5, "raw_sw_index": 6,
                         "basin_info_path": manifest_path, "context_dim": 8, "dim": 32,
                         "depth": 1, "snow": True},
        "dataset_params": {
            "class": "MultiBasinCatchmentWindow",
            "training_path": manifest_path, "validation_path": manifest_path,
            "test_path": manifest_path,
            "batch_size": 4, "forecast_history": 120, "forecast_length": 48,
            "target_col": ["cfs"], "relevant_cols": RELEVANT_COLS,
            "scaled_cols": SCALED_COLS, "window_stride": 96,
            "train_basin_split": "train", "valid_basin_split": "train",
            "test_basin_split": "train",
            "train_end_date": TRAIN_END,
            "valid_start_date": TRAIN_END, "valid_end_date": VALID_END,
            "test_start_date": VALID_END,
            "train_samples_per_epoch": 24,
        },
        "training_params": {"criterion": "MSE", "optimizer": "Adam", "optim_params": {},
                            "lr": 0.003, "epochs": 1, "batch_size": 4},
        "GCS": False, "wandb": False, "forward_params": {}, "metrics": ["MSE"],
    }


class TestMultiBasinWindowLoader(unittest.TestCase):
    """Tests for the multi-basin loader built from a synthetic manifest."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.manifest_path = build_manifest(cls.tmp.name)
        cls.params = make_params(cls.manifest_path)
        cls.train_loader = MultiBasinWindowLoader(
            cls.manifest_path, 120, 48, ["cfs"], RELEVANT_COLS, scaled_cols=SCALED_COLS,
            end_date=TRAIN_END, basin_split="train", window_stride=96, samples_per_epoch=24)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_shapes_and_basin_channel(self):
        src, trg = self.train_loader[0]
        self.assertEqual(src.shape, (168, len(RELEVANT_COLS) + 1))
        self.assertEqual(trg.shape, (48, len(RELEVANT_COLS) + 1))
        self.assertTrue((trg[:, -1] == src[0, -1]).all())
        # The basin channel is constant and holds the manifest position.
        self.assertTrue((src[:, -1] == src[0, -1]).all())
        self.assertEqual(src[0, -1].item(), 0.0)
        last = self.train_loader[len(self.train_loader) - 1][0]
        self.assertEqual(last[0, -1].item(), 1.0)
        # Horizon flow zeroed in src (no leakage), finite everywhere.
        self.assertEqual(src[120:, 0].abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(src).all())

    def test_target_standardization(self):
        basin, local = self.train_loader.locate(0)
        raw_src, raw_trg = self.train_loader.basin_loaders[basin][local]
        _, trg = self.train_loader[0]
        scale = self.train_loader.flow_scales[basin]
        self.assertTrue(torch.allclose(trg[:, 0] * scale, raw_trg[:, 0], atol=1e-6))
        # Spin-up flow in src stays physical mm/hr.
        src, _ = self.train_loader[0]
        self.assertTrue(torch.allclose(src[:120, 0], raw_src[:120, 0]))

    def test_split_selection(self):
        holdout = MultiBasinWindowLoader(
            self.manifest_path, 120, 48, ["cfs"], RELEVANT_COLS, scaled_cols=SCALED_COLS,
            basin_split="holdout", window_stride=96)
        self.assertEqual(holdout.basin_site_ids, ["basinC"])
        self.assertEqual(holdout.basin_positions, [2])

    def test_pretrained_embedding_filter(self):
        embedded_train = MultiBasinWindowLoader(
            self.manifest_path, 120, 48, ["cfs"], RELEVANT_COLS, scaled_cols=SCALED_COLS,
            basin_split="train", window_stride=96, require_pretrained_embedding=True)
        self.assertEqual(embedded_train.basin_site_ids, ["basinA"])
        self.assertEqual(embedded_train.basin_positions, [0])

    def test_sample_weights(self):
        weights = self.train_loader.sample_weights
        self.assertEqual(len(weights), len(self.train_loader))
        self.assertTrue((weights > 0).all())
        self.assertEqual(self.train_loader.samples_per_epoch, 24)

    def test_p01m_filled_from_precip(self):
        # Windows in the first third of the record (NaN p01m) must still exist and be finite.
        idx = RELEVANT_COLS.index("p01m")
        src, _ = self.train_loader[0]
        self.assertTrue(torch.isfinite(src[:, idx]).all())


class TestHybridGR4MultiBasin(unittest.TestCase):
    """Tests for the multi-basin model wrapper."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.manifest_path = build_manifest(cls.tmp.name)
        torch.manual_seed(0)
        cls.model = HybridGR4MultiBasin(len(RELEVANT_COLS) + 1, 120, 48, 5, 6,
                                        cls.manifest_path, context_dim=8, dim=32, depth=1,
                                        snow=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_context_selection(self):
        bank = torch.load(os.path.join(self.tmp.name, "embeddings.pt"), weights_only=True)
        context = self.model.basin_context(torch.tensor([0, 1, 2]))
        self.assertTrue(torch.allclose(context[0], bank["embeddings"][0]))
        self.assertTrue(torch.allclose(context[2], bank["embeddings"][1]))
        self.assertTrue(torch.allclose(context[1], self.model.learned_context.weight[1]))

    def test_forward_shape_and_scaling(self):
        loader = MultiBasinWindowLoader(self.manifest_path, 120, 48, ["cfs"], RELEVANT_COLS,
                                        scaled_cols=SCALED_COLS, basin_split="train",
                                        window_stride=96, end_date=TRAIN_END)
        src = torch.stack([loader[0][0], loader[len(loader) - 1][0]])
        out = self.model(src)
        self.assertEqual(out.shape, (2, 48))
        self.assertTrue(torch.isfinite(out).all())


class TestSweSeeding(unittest.TestCase):
    """Tests for the observed-SWE channel and snow-store seeding."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.manifest_path = build_manifest(cls.tmp.name, with_swe=True)
        cls.swe_index = SWE_RELEVANT_COLS.index("snodas_swe_mm")
        cls.loader = MultiBasinWindowLoader(
            cls.manifest_path, 120, 48, ["cfs"], SWE_RELEVANT_COLS, scaled_cols=SCALED_COLS,
            basin_split="train", window_stride=96, end_date=TRAIN_END)
        torch.manual_seed(0)
        cls.model = HybridGR4MultiBasin(len(SWE_RELEVANT_COLS) + 1, 120, 48, 5, 6,
                                        cls.manifest_path, context_dim=8, dim=32, depth=1,
                                        snow=True, match_flow=False,
                                        swe_index=cls.swe_index)
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_swe_channel_values_and_sentinel(self):
        # Record starts 2022-01-01; the first window's spin-up start is inside SWE coverage.
        src, _ = self.loader[0]
        self.assertAlmostEqual(src[0, self.swe_index].item(), SWE_TEST_VALUE, places=4)
        # Windows past the coverage end must carry the sentinel instead.
        basin, _ = self.loader.locate(0)
        loader_a = self.loader.basin_loaders[basin]
        starts = loader_a.valid_starts
        cutoff = (pd.Timestamp(SWE_COVERAGE_END) - pd.Timestamp("2022-01-01")).days * 24 + 24
        late = next(i for i, s in enumerate(starts) if s >= cutoff)
        late_src, _ = self.loader[late]
        self.assertEqual(late_src[0, self.swe_index].item(), -1.0)

    def test_missing_series_yields_sentinel(self):
        holdout = MultiBasinWindowLoader(
            self.manifest_path, 120, 48, ["cfs"], SWE_RELEVANT_COLS, scaled_cols=SCALED_COLS,
            basin_split="holdout", window_stride=96)
        src, _ = holdout[0]
        self.assertTrue((src[:, self.swe_index] == -1.0).all())

    def test_met_excludes_swe_and_flow(self):
        self.assertNotIn(self.swe_index, self.model.met_indices)
        self.assertNotIn(0, self.model.met_indices)

    def test_seeded_snow_raises_flow(self):
        src, _ = self.loader[0]
        batch = src.unsqueeze(0)
        sentinel = batch.clone()
        sentinel[:, :, self.swe_index] = -1.0
        seeded = batch.clone()
        seeded[:, :, self.swe_index] = 400.0
        with torch.no_grad():
            flow_sentinel = self.model(sentinel)
            flow_seeded = self.model(seeded)
        # Warm synthetic temps melt the seeded pack, so simulated flow must strictly increase.
        self.assertGreater(flow_seeded.sum().item(), flow_sentinel.sum().item())

    def test_sentinel_equals_zero_seed(self):
        src, _ = self.loader[0]
        batch = src.unsqueeze(0)
        sentinel = batch.clone()
        sentinel[:, :, self.swe_index] = -1.0
        zero = batch.clone()
        zero[:, :, self.swe_index] = 0.0
        with torch.no_grad():
            self.assertTrue(torch.allclose(self.model(sentinel), self.model(zero),
                                           atol=1e-6))


class TestAnchoredForcing(unittest.TestCase):
    """Tests for the station-observation mask and the anchored (physically driven) forcing path."""

    PHYS_COLS = ["precip_raw", "pet_raw", "asos_raw", "asos_observed"]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.manifest_path = build_manifest(cls.tmp.name)
        cls.cols = RELEVANT_COLS + cls.PHYS_COLS
        cls.phys_indices = [cls.cols.index(c) for c in cls.PHYS_COLS]
        cls.loader = MultiBasinWindowLoader(
            cls.manifest_path, 120, 48, ["cfs"], cls.cols, scaled_cols=SCALED_COLS,
            basin_split="train", window_stride=96, end_date=TRAIN_END)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_mask_records_pre_fill_availability(self):
        # make_gauge_frame blanks p01m over the first third of the record, and the loader fills it
        # from NLDAS precip -- the mask must still report those hours as unobserved.
        src, _ = self.loader[0]
        mask = src[:, self.cols.index("asos_observed")]
        self.assertTrue(((mask == 0.0) | (mask == 1.0)).all())
        self.assertEqual(mask[0].item(), 0.0)
        late = self.loader[len(self.loader) - 1][0][:, self.cols.index("asos_observed")]
        self.assertEqual(late[-1].item(), 1.0)

    def test_physical_channels_are_unscaled(self):
        src, _ = self.loader[0]
        raw_precip = src[:, self.cols.index("precip_raw")]
        scaled_precip = src[:, self.cols.index("precipitation")]
        self.assertTrue((raw_precip >= 0).all())          # physical mm/hr, never negative
        self.assertFalse(torch.allclose(raw_precip, scaled_precip))

    def _model(self, **kwargs):
        torch.manual_seed(0)
        return HybridGR4MultiBasin(len(self.cols) + 1, 120, 48, self.cols.index("temp_lapse_k"),
                                   self.cols.index("sw_raw"), self.manifest_path, context_dim=8,
                                   dim=32, depth=1, snow=True, match_flow=False,
                                   phys_indices=self.phys_indices, anchored=True, **kwargs)

    def test_physical_channels_excluded_from_encoder(self):
        model = self._model()
        for index in self.phys_indices:
            self.assertNotIn(index, model.met_indices)

    def test_multiplier_starts_at_physics_baseline(self):
        # The multiplier is initialised near (not exactly at) 1.0 so the encoder has a live
        # gradient path; the multiplier-only model therefore starts within a fraction of a percent
        # of the pure-physics run. The ASOS gate is deliberately NOT asserted to be inert here --
        # that would reward initialising it so small it cannot train.
        src = torch.stack([self.loader[0][0], self.loader[1][0]])
        with torch.no_grad():
            physics = self._model(use_multiplier=False, use_asos_gate=False)(src)
            multiplier = self._model(use_multiplier=True, use_asos_gate=False)(src)
        self.assertTrue(torch.isfinite(multiplier).all())
        scale = physics.abs().max().clamp(min=1e-8)
        self.assertLess(float((multiplier - physics).abs().max() / scale), 0.02)

    def test_asos_only_storm_is_admitted_at_init(self):
        # A storm the grid missed entirely must produce a materially non-zero effective rainfall
        # from the start, otherwise the station pathway is untrainable in short runs.
        model = self._model(use_multiplier=False, use_asos_gate=True)
        gen = model.hybrid.forcing_generator
        steps, storm = 48, 5.0
        phys = torch.zeros(1, steps, 4)
        phys[..., 2] = storm          # station reports rain
        phys[..., 3] = 1.0            # and was genuinely observing
        with torch.no_grad():
            p_eff = gen(torch.zeros(1, steps, len(model.met_indices)),
                        torch.zeros(1, 8), phys_forcing=phys)[..., 0]
        recovered = float(p_eff.mean()) / storm
        self.assertGreater(recovered, 0.05)   # not driven to irrelevance by the init
        self.assertLess(recovered, 1.0)       # and still a bounded fraction of the point value

    def _window_with_rain(self):
        """
        Returns the first window whose horizon actually contains gridded precipitation.

        :return: A source tensor of shape (1, spinup + horizon, n_features + 1).
        :rtype: torch.Tensor
        """
        column = self.cols.index("precip_raw")
        for index in range(len(self.loader)):
            src = self.loader[index][0]
            if float(src[120:, column].sum()) > 0:
                return src.unsqueeze(0)
        self.skipTest("fixture contains no window with horizon rainfall")

    def _encoder_grads(self, model):
        """
        Returns the finite gradient magnitudes of the forcing generator's encoder parameters.

        :param model: A model on which backward() has already been called.
        :type model: torch.nn.Module
        :return: The per-parameter maximum absolute gradients.
        :rtype: list
        """
        return [float(p.grad.abs().max()) for n, p in model.named_parameters()
                if "forcing_generator.encoder" in n and p.grad is not None]

    def test_gradients_reach_the_encoder_when_it_rains(self):
        model = self._model()
        model(self._window_with_rain()).sum().backward()
        grads = self._encoder_grads(model)
        self.assertTrue(grads)
        self.assertGreater(max(grads), 0.0)

    def test_dry_window_gives_the_encoder_no_gradient(self):
        # Structural consequence of a multiplicative anchor: d(P_eff)/d(multiplier) = P_grid, so a
        # rain-free window carries no signal for the PRECIPITATION CORRECTION specifically. This is
        # correct attribution rather than lost training: the GR4/snow parameters still receive
        # gradient from the same window through recession, routing and melt, which is precisely the
        # hybrid's advantage over a pure sequence model -- recession comes from the ODE for free
        # instead of having to be learned from data.
        model = self._model()
        src = torch.stack([self.loader[0][0]])
        column = self.cols.index("precip_raw")
        if float(src[0, :, column].sum()) > 0:
            self.skipTest("first fixture window is not dry")
        model(src).sum().backward()
        self.assertEqual(max(self._encoder_grads(model)), 0.0)


class TestGapPolicy(unittest.TestCase):
    """Tests that interpolation respects the channel policy and the true gap limit."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.manifest_path = build_manifest(cls.tmp.name)
        with open(cls.manifest_path) as f:
            manifest = json.load(f)
        manifest["preprocessing"]["no_interp_cols"] = ["precipitation", "p01m", "precip_raw",
                                                       "asos_raw"]
        # Punch known gaps into one basin: 6 hours (fillable) and 7 hours (not) in temperature,
        # and a 3-hour hole in precipitation, which must never be filled.
        path = manifest["basins"][0]["csv_path"]
        frame = pd.read_csv(path)
        frame.loc[3000:3005, "temperature"] = np.nan
        frame.loc[4000:4006, "temperature"] = np.nan
        frame.loc[5000:5002, "precipitation"] = np.nan
        frame.to_csv(path, index=False)
        with open(cls.manifest_path, "w") as f:
            json.dump(manifest, f)
        cls.cols = RELEVANT_COLS + ["precip_raw", "pet_raw", "asos_raw", "asos_observed"]
        cls.loader = MultiBasinWindowLoader(
            cls.manifest_path, 120, 48, ["cfs"], cls.cols, scaled_cols=SCALED_COLS,
            basin_split="train", window_stride=24, max_basins=1, max_input_gap=6)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_six_hour_gap_filled_seven_hour_gap_not(self):
        frame = self.loader.basin_loaders[0].df
        self.assertTrue(np.isfinite(frame["temperature"].to_numpy()[3000:3006]).all())
        self.assertFalse(np.isfinite(frame["temperature"].to_numpy()[4000:4007]).all())

    def test_precipitation_never_interpolated(self):
        frame = self.loader.basin_loaders[0].df
        self.assertFalse(np.isfinite(frame["precipitation"].to_numpy()[5000:5003]).any())

    def test_served_windows_are_all_finite(self):
        for index in range(0, len(self.loader), max(1, len(self.loader) // 20)):
            src, trg = self.loader[index]
            self.assertTrue(torch.isfinite(src).all())
            self.assertTrue(torch.isfinite(trg).all())


class TestMultiBasinEndToEnd(unittest.TestCase):
    """End-to-end: multi-basin training through FF's config-driven pipeline."""

    def test_config_driven_training(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        manifest_path = build_manifest(tmp.name)
        params = make_params(manifest_path)
        model = PyTorchForecast(params["model_name"], manifest_path, manifest_path,
                                manifest_path, params)
        self.assertGreater(len(model.training), 20)
        self.assertGreater(len(model.validation), 4)
        train_transformer_style(model, params["training_params"], forward_params={})


if __name__ == "__main__":
    unittest.main()
