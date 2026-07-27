"""
PyTorch dataset for catchment embedding records (.npz files produced by Water's embedding_dataset.py).

Each record holds one site's Sentinel patch, static attribute vector and multi-year daily flow
history. The dataset normalizes all three modalities and, per access, samples a random fixed-length
history window — so every epoch shows the encoder a different slice of each site's behavioral record,
a natural augmentation for contrastive pretraining.

History is returned as two channels: standardized log-flow and an observed mask (1 where the gauge
reported), so missing stretches are explicit rather than imputed silently.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CatchmentEmbeddingDataset(Dataset):
    """
    Loads per-site .npz embedding records and serves normalized (image, static, history) tensors.
    """

    def __init__(self, data_dir: str, history_window_days: int = 365,
                 image_scale: float = 3000.0, min_window_observed: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initializes the dataset and computes normalization statistics across sites.

        :param data_dir: Directory containing the <site>.npz records.
        :type data_dir: str
        :param history_window_days: The length of the sampled history window, defaults to 365.
        :type history_window_days: int, optional
        :param image_scale: Reflectance divisor for the imagery (Sentinel-2 L1C DN), defaults
            to 3000.0.
        :type image_scale: float, optional
        :param min_window_observed: Minimum observed fraction for a sampled window before falling
            back to the most-observed window, defaults to 0.5.
        :type min_window_observed: float, optional
        :param seed: Optional seed for reproducible window sampling, defaults to None.
        :type seed: int, optional
        """
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
                history = record["history"]
                log_flows.append(np.log1p(np.clip(history[np.isfinite(history)], 0.0, None)))
        static_matrix = np.stack(statics)
        self.static_mean = np.nanmean(static_matrix, axis=0)
        self.static_std = np.nanstd(static_matrix, axis=0)
        self.static_std[self.static_std == 0] = 1.0
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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns one site's normalized modalities.

        :param index: The site index.
        :type index: int
        :return: A dict with "image" (bands, H, W), "static" (static_features,), "history"
            (window, 2) [standardized log-flow, observed mask] and "site_index" tensors.
        :rtype: Dict[str, torch.Tensor]
        """
        record = np.load(os.path.join(self.data_dir, self.site_ids[index] + ".npz"))
        image = np.clip(record["image"] / self.image_scale, 0.0, 2.0).astype(np.float32)

        static = (record["static"] - self.static_mean) / self.static_std
        static = np.nan_to_num(static, nan=0.0).astype(np.float32)

        window = self._sample_history_window(record["history"])
        observed = np.isfinite(window)
        log_flow = np.zeros_like(window, dtype=np.float32)
        log_flow[observed] = (np.log1p(np.clip(window[observed], 0.0, None)) -
                              self.flow_mean) / self.flow_std
        history = np.stack([log_flow, observed.astype(np.float32)], axis=-1)

        return {"image": torch.from_numpy(image), "static": torch.from_numpy(static),
                "history": torch.from_numpy(history),
                "site_index": torch.tensor(index, dtype=torch.long)}
