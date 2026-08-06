"""
Sliding-window dataset for the operational forecasting task: hourly flow, 14 days out.

Each sample is a (spin-up, horizon) pair cut from a gauge's hourly record: the spin-up segment
(observed meteorology + flow) is used to estimate the model state at forecast time t0, and the
horizon segment carries the "forecast" meteorology (observed met during training — the
perfect-prognosis convention) and the target flow. Windows with too much missing data are skipped
at index-build time so training never sees unusable pairs.

Flow is converted from cfs to mm per hour over the basin (the ODE's native unit) using the drainage
area; met channels are z-scored with stats computed over the training portion of the record, except
precipitation channels, which stay physical (mm) because they are water inputs, not covariates.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CFS_TO_MM_HR_PER_KM2 = 0.0283168 * 3.6


class ForecastWindowDataset(Dataset):
    """
    Serves (spin-up, horizon) hourly windows from one gauge's scraped record.
    """

    def __init__(self, frame: pd.DataFrame, met_columns: Sequence[str], area_km2: float,
                 raw_temp_column: str = "temperature", raw_sw_column: str = "shortwave_radiation",
                 target_column: str = "cfs", spinup_hours: int = 720, horizon_hours: int = 336,
                 stride_hours: int = 24, physical_columns: Sequence[str] = ("precipitation", "p01m"),
                 temp_offset_c: float = 0.0, min_valid_fraction: float = 0.95,
                 norm_stats: Optional[Dict[str, np.ndarray]] = None):
        """
        Initializes the dataset and builds the valid-window index.

        :param frame: The gauge's hourly dataframe with a "datetime" column (or DatetimeIndex).
        :type frame: pd.DataFrame
        :param met_columns: The meteorology columns fed to the forcing generator.
        :type met_columns: Sequence[str]
        :param area_km2: The basin drainage area in km^2 for the cfs-to-mm conversion.
        :type area_km2: float
        :param raw_temp_column: The temperature column (Kelvin) for the snow physics channel,
            defaults to "temperature".
        :type raw_temp_column: str, optional
        :param raw_sw_column: The shortwave column for the snow physics channel, defaults to
            "shortwave_radiation".
        :type raw_sw_column: str, optional
        :param target_column: The flow column in cfs, defaults to "cfs".
        :type target_column: str, optional
        :param spinup_hours: The state-estimation window length, defaults to 720 (30 days).
        :type spinup_hours: int, optional
        :param horizon_hours: The forecast window length, defaults to 336 (14 days).
        :type horizon_hours: int, optional
        :param stride_hours: The spacing between consecutive forecast issue times, defaults to 24.
        :type stride_hours: int, optional
        :param physical_columns: Met columns kept in physical units (not z-scored), defaults to the
            precipitation channels.
        :type physical_columns: Sequence[str], optional
        :param temp_offset_c: Additive lapse-rate correction applied to the raw temperature channel
            in degC (basin-mean minus measurement elevation times lapse rate), defaults to 0.0.
        :type temp_offset_c: float, optional
        :param min_valid_fraction: Minimum observed fraction of flow and met required in both
            segments for a window to be indexed, defaults to 0.95.
        :type min_valid_fraction: float, optional
        :param norm_stats: Optional precomputed normalization stats ("mean"/"std" arrays over
            met_columns) so validation/test sets reuse the training statistics, defaults to None
            which computes them from this frame.
        :type norm_stats: Dict[str, np.ndarray], optional
        """
        if "datetime" in frame.columns:
            frame = frame.set_index("datetime")
        frame = frame.sort_index()
        self.met_columns = list(met_columns)
        self.spinup_hours = spinup_hours
        self.horizon_hours = horizon_hours

        met = frame[self.met_columns].to_numpy(dtype=np.float32)
        flow_mm = (frame[target_column].to_numpy(dtype=np.float32) *
                   CFS_TO_MM_HR_PER_KM2 / area_km2)
        temp_c = frame[raw_temp_column].to_numpy(dtype=np.float32) - 273.15 + temp_offset_c
        shortwave = frame[raw_sw_column].to_numpy(dtype=np.float32)

        if norm_stats is None:
            mean = np.nanmean(met, axis=0)
            std = np.nanstd(met, axis=0)
            std[std == 0] = 1.0
            norm_stats = {"mean": mean, "std": std}
        self.norm_stats = norm_stats
        normalized = (met - norm_stats["mean"]) / norm_stats["std"]
        for i, name in enumerate(self.met_columns):
            if name in physical_columns:
                normalized[:, i] = met[:, i]
        self.met = np.nan_to_num(normalized, nan=0.0)
        self.flow_mm = flow_mm
        self.raw = np.nan_to_num(np.stack([temp_c, shortwave], axis=-1), nan=0.0)
        self.timestamps = frame.index

        observed = np.isfinite(flow_mm) & np.isfinite(met).all(axis=1)
        window = spinup_hours + horizon_hours
        self.starts: List[int] = []
        for start in range(0, len(frame) - window + 1, stride_hours):
            if observed[start:start + window].mean() >= min_valid_fraction:
                self.starts.append(start)

    def __len__(self) -> int:
        """
        Returns the number of forecast windows.

        :return: The window count.
        :rtype: int
        """
        return len(self.starts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns one (spin-up, horizon) pair.

        :param index: The window index.
        :type index: int
        :return: A dict with "spinup_met", "spinup_flow", "spinup_raw", "horizon_met",
            "horizon_flow", "horizon_raw" tensors and the "t0" position (int tensor); flows are in
            mm/hr, NaNs already replaced (validity guaranteed by the index filter).
        :rtype: Dict[str, torch.Tensor]
        """
        start = self.starts[index]
        t0 = start + self.spinup_hours
        end = t0 + self.horizon_hours
        return {
            "spinup_met": torch.from_numpy(self.met[start:t0]),
            "spinup_flow": torch.from_numpy(np.nan_to_num(self.flow_mm[start:t0], nan=0.0)),
            "spinup_raw": torch.from_numpy(self.raw[start:t0]),
            "horizon_met": torch.from_numpy(self.met[t0:end]),
            "horizon_flow": torch.from_numpy(np.nan_to_num(self.flow_mm[t0:end], nan=0.0)),
            "horizon_raw": torch.from_numpy(self.raw[t0:end]),
            "t0": torch.tensor(t0, dtype=torch.long),
        }
