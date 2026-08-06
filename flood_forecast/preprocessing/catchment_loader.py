"""
PyTorch dataset for catchment embedding records (.npz files produced by Water's embedding_dataset.py).

Each record holds one site's Sentinel patch, static attribute vector and multi-year daily flow
history. The dataset normalizes all three modalities and serves the history in one of two modes:

``random_window`` (legacy): per access, samples a random fixed-length daily window standardized
with GLOBAL cross-site flow statistics. Diagnosed failure mode (2026-08): global standardization
lets absolute magnitude dominate the contrastive task (the embedding collapses onto basin scale),
random windows can miss regime-defining years, and windows carry no calendar anchor so
seasonality timing is scrambled.

``hourly_panel``: deterministic — six 92-day HOURLY slices per site (four best-observed calendar
seasons plus the record's flood and drought windows, built by Water's build_panel_records.py),
per-SITE standardized so only regime shape (not size) remains. Each timestep carries
day-of-year encodings as the calendar anchor plus flood/drought slice-type flags, and hourly
resolution preserves the flash dynamics that the daily legacy records could not represent.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CatchmentEmbeddingDataset(Dataset):
    """
    Loads per-site .npz embedding records and serves normalized (image, static, history) tensors.
    """

    def __init__(self, data_dir: str, history_window_days: int = 365,
                 image_scale: float = 3000.0, min_window_observed: float = 0.5,
                 seed: Optional[int] = None, history_mode: str = "random_window",
                 cross_year_views: bool = False):
        """
        Initializes the dataset and computes normalization statistics across sites.

        :param data_dir: Directory containing the <site>.npz records (legacy daily records for
            ``random_window``, panel records for ``hourly_panel``).
        :type data_dir: str
        :param history_window_days: The length of the sampled history window (``random_window``
            mode only), defaults to 365.
        :type history_window_days: int, optional
        :param image_scale: Reflectance divisor for the imagery (Sentinel-2 L1C DN), defaults
            to 3000.0.
        :type image_scale: float, optional
        :param min_window_observed: Minimum observed fraction for a sampled window before falling
            back to the most-observed window (``random_window`` mode only), defaults to 0.5.
        :type min_window_observed: float, optional
        :param seed: Optional seed for reproducible window sampling, defaults to None.
        :type seed: int, optional
        :param history_mode: "random_window" (legacy daily) or "hourly_panel" (deterministic
            seasonal/extreme hourly slices), defaults to "random_window".
        :type history_mode: str, optional
        :param cross_year_views: hourly_panel only — items additionally carry "history_alt", a
            second view whose seasonal slices come from different years than "history", for
            cross-year positive pairs; extraction should leave this False so the canonical
            best-coverage panel is used deterministically. Defaults to False.
        :type cross_year_views: bool, optional
        """
        if history_mode not in ("random_window", "hourly_panel"):
            raise ValueError("history_mode must be 'random_window' or 'hourly_panel'")
        if cross_year_views and history_mode != "hourly_panel":
            raise ValueError("cross_year_views requires history_mode='hourly_panel'")
        self.history_mode = history_mode
        self.cross_year_views = cross_year_views
        self.data_dir = data_dir
        self.history_window_days = history_window_days
        self.image_scale = image_scale
        self.min_window_observed = min_window_observed
        self.rng = np.random.default_rng(seed)
        self.site_ids: List[str] = sorted(
            name[:-4] for name in os.listdir(data_dir) if name.endswith(".npz"))
        if not self.site_ids:
            raise ValueError("No .npz records found in " + data_dir)

        statics, log_flows = [], []
        for site_id in self.site_ids:
            with np.load(os.path.join(data_dir, site_id + ".npz")) as record:
                statics.append(record["static"])
                if self.history_mode == "random_window":
                    history = record["history"]
                    log_flows.append(np.log1p(np.clip(history[np.isfinite(history)], 0.0,
                                                      None)))
        static_matrix = np.stack(statics)
        self.static_mean = np.nanmean(static_matrix, axis=0)
        self.static_std = np.nanstd(static_matrix, axis=0)
        self.static_std[self.static_std == 0] = 1.0
        if self.history_mode == "random_window":
            # Global cross-site standardization (legacy); hourly_panel standardizes per site
            # inside __getitem__ so magnitude cannot dominate the contrastive task.
            all_log_flow = np.concatenate(log_flows)
            self.flow_mean = float(all_log_flow.mean())
            self.flow_std = float(all_log_flow.std()) or 1.0
        self.static_features = static_matrix.shape[1]

    def __len__(self) -> int:
        """
        Returns the number of sites.

        :return: The site count.
        :rtype: int
        """
        return len(self.site_ids)

    def _sample_history_window(self, history: np.ndarray) -> np.ndarray:
        """
        Samples a random window of the history with sufficient observed data.

        :param history: The full daily history with NaNs for missing days.
        :type history: np.ndarray
        :return: A window of length history_window_days.
        :rtype: np.ndarray
        """
        window = self.history_window_days
        max_start = len(history) - window
        if max_start <= 0:
            padded = np.full(window, np.nan, dtype=history.dtype)
            padded[-len(history):] = history
            return padded
        for _ in range(8):
            start = int(self.rng.integers(0, max_start + 1))
            candidate = history[start:start + window]
            if np.isfinite(candidate).mean() >= self.min_window_observed:
                return candidate
        observed = np.isfinite(history).astype(float)
        coverage = np.convolve(observed, np.ones(window), mode="valid")
        best_start = int(coverage.argmax())
        return history[best_start:best_start + window]

    def _choose_panel_rows(self, types: List[str],
                           avoid: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        """
        Picks one stored slice row per slice type (season/flood/drought).

        :param types: The per-row slice type labels (seasons may repeat across years).
        :type types: List[str]
        :param avoid: Optional type -> row mapping of a sibling view; where a type has
            alternatives, a different year is chosen so the two views share no seasonal slice.
        :type avoid: Dict[str, int], optional
        :return: Slice type -> chosen row index.
        :rtype: Dict[str, int]
        """
        rows_by_type: Dict[str, List[int]] = {}
        for row, slice_type in enumerate(types):
            rows_by_type.setdefault(slice_type, []).append(row)
        chosen = {}
        for slice_type, rows in rows_by_type.items():
            if not self.cross_year_views:
                chosen[slice_type] = rows[0]  # coverage-best canonical year
                continue
            options = rows
            if avoid is not None and len(rows) > 1:
                options = [r for r in rows if r != avoid.get(slice_type)] or rows
            chosen[slice_type] = options[int(self.rng.integers(0, len(options)))]
        return chosen

    def _panel_history(self, record,
                       avoid: Optional[Dict[str, int]] = None
                       ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Builds one per-site standardized hourly-panel history view.

        :param record: An open panel .npz record with panel/panel_types/panel_starts.
        :type record: numpy.lib.npyio.NpzFile
        :param avoid: Optional sibling view's type -> row choices to differ from (cross-year
            positive views), defaults to None.
        :type avoid: Dict[str, int], optional
        :return: A tuple of (history of shape (n_types, slice_len, 6) with channels
            [standardized log-flow, observed mask, sin day-of-year, cos day-of-year,
            flood flag, drought flag], the chosen type -> row mapping).
        :rtype: Tuple[np.ndarray, Dict[str, int]]
        """
        all_types = [str(t) for t in record["panel_types"]]
        chosen = self._choose_panel_rows(all_types, avoid=avoid)
        ordered_types = list(dict.fromkeys(all_types))
        if self.cross_year_views:
            # Training views are seasonal-only: the flood/drought slices are single-year, so
            # keeping them would hand both views an identical (and highly distinctive) slice —
            # a memorization shortcut that defeats the cross-year invariance objective.
            ordered_types = [t for t in ordered_types if t not in ("flood", "drought")]
        rows = [chosen[t] for t in ordered_types]  # stable type order
        full_panel = record["panel"].astype(np.float32)
        panel = full_panel[rows]
        observed = np.isfinite(panel)
        log_flow = np.log1p(np.clip(np.nan_to_num(panel, nan=0.0), 0.0, None))
        # Standardization statistics come from ALL stored slices so both cross-year views of a
        # site are normalized identically (view-dependent stats would leak view identity).
        full_observed = np.isfinite(full_panel)
        site_values = np.log1p(np.clip(np.nan_to_num(full_panel, nan=0.0), 0.0,
                                       None))[full_observed]
        site_mean = float(site_values.mean()) if site_values.size else 0.0
        site_std = float(site_values.std()) or 1.0
        standardized = np.where(observed, (log_flow - site_mean) / site_std, 0.0)

        n_slices, slice_len = panel.shape
        day_of_year = np.empty((n_slices, slice_len), dtype=np.float32)
        starts = [str(record["panel_starts"][row]) for row in rows]
        for position, start in enumerate(starts):
            start_time = pd.Timestamp(start)
            hours = start_time.dayofyear * 24.0 + start_time.hour + np.arange(slice_len)
            day_of_year[position] = (hours % (365.25 * 24)) / (365.25 * 24)
        selected_types = [all_types[row] for row in rows]
        flood = np.array([t == "flood" for t in selected_types],
                         dtype=np.float32)[:, None] * np.ones_like(standardized)
        drought = np.array([t == "drought" for t in selected_types],
                           dtype=np.float32)[:, None] * np.ones_like(standardized)
        history = np.stack([standardized, observed.astype(np.float32),
                            np.sin(2 * np.pi * day_of_year), np.cos(2 * np.pi * day_of_year),
                            flood, drought], axis=-1).astype(np.float32)
        return history, chosen

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns one site's normalized modalities.

        :param index: The site index.
        :type index: int
        :return: A dict with "image" (bands, H, W), "static" (static_features,), "history"
            ((window, 2) [standardized log-flow, observed mask] in ``random_window`` mode,
            (n_slices, slice_len, 6) in ``hourly_panel`` mode) and "site_index" tensors.
        :rtype: Dict[str, torch.Tensor]
        """
        record = np.load(os.path.join(self.data_dir, self.site_ids[index] + ".npz"))
        image = np.clip(record["image"] / self.image_scale, 0.0, 2.0).astype(np.float32)

        static = (record["static"] - self.static_mean) / self.static_std
        static = np.nan_to_num(static, nan=0.0).astype(np.float32)

        if self.history_mode == "hourly_panel":
            history, chosen = self._panel_history(record)
            item = {"image": torch.from_numpy(image), "static": torch.from_numpy(static),
                    "history": torch.from_numpy(history),
                    "site_index": torch.tensor(index, dtype=torch.long)}
            if self.cross_year_views:
                alt, _ = self._panel_history(record, avoid=chosen)
                item["history_alt"] = torch.from_numpy(alt)
            return item
        window = self._sample_history_window(record["history"])
        observed = np.isfinite(window)
        log_flow = np.zeros_like(window, dtype=np.float32)
        log_flow[observed] = (np.log1p(np.clip(window[observed], 0.0, None)) -
                              self.flow_mean) / self.flow_std
        history = np.stack([log_flow, observed.astype(np.float32)], axis=-1)

        return {"image": torch.from_numpy(image), "static": torch.from_numpy(static),
                "history": torch.from_numpy(history),
                "site_index": torch.tensor(index, dtype=torch.long)}
