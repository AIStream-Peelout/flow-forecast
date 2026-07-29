from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import torch
from typing import Dict, Tuple, Union, Optional, List
from flood_forecast.pre_dict import interpolate_dict
from flood_forecast.preprocessing.buil_dataset import get_data
from datetime import datetime
from flood_forecast.preprocessing.temporal_feats import feature_fix
from copy import deepcopy


def to_tz_naive_datetime(series: pd.Series) -> pd.Series:
    """
    Converts a column of timestamps to timezone-naive ``datetime64[ns]``.

    Some data sources (e.g. USGS/NOAA feeds) provide timezone-aware timestamps such as
    ``2014-04-11 16:00:00+00:00``. Casting these directly with ``astype("datetime64[ns]")``
    raises ``ValueError: cannot supply both a tz and a timezone-naive dtype``. This helper
    parses to UTC and drops the timezone so downstream sorting and indexing behave the same
    whether the input is tz-aware or tz-naive. For already tz-naive input the wall-clock values
    are unchanged.

    :param series: A pandas Series of datetime-like values (strings or datetimes).
    :type series: pd.Series
    :return: The series converted to timezone-naive ``datetime64[ns]``.
    :rtype: pd.Series
    """
    return pd.to_datetime(series, utc=True).dt.tz_localize(None).astype("datetime64[ns]")


class CSVDataLoader(Dataset):
    """
    A base data loader that takes a CSV file and properly batches time series
    data for use in training or evaluating a PyTorch model.
    """
    def __init__(
        self,
        file_path: str,
        forecast_history: int,
        forecast_length: int,
        target_col: List,
        relevant_cols: List,
        scaling=None,
        start_stamp: int = 0,
        end_stamp: int = None,
        gcp_service_key: Optional[str] = None,
        interpolate_param: bool = False,
        sort_column=None,
        scaled_cols=None,
        feature_params=None,
        no_scale=False,
        preformatted_df=False

    ):
        """
        Initializes the CSVDataLoader.

        :param file_path: The path to the CSV file you wish to use (GCS compatible) or a Pandas dataframe.
        :type file_path: str
        :param forecast_history: The length of the historical time series data you wish to
                                 utilize for forecasting (encoder input length).
        :type forecast_history: int
        :param forecast_length: The number of time steps to forecast ahead (decoder output length).
        :type forecast_length: int
        :param relevant_cols: Supply column names you wish to use as features (including the target column).
        :type relevant_cols: List
        :param target_col: The target column or columns you to predict. Must be a list, e.g., ['cfs'].
        :type target_col: List
        :param scaling: (Highly recommended) If provided, should be a subclass of ``sklearn.base.BaseEstimator``
                        and ``sklearn.base.TransformerMixin`` (i.e., StandardScaler, MaxAbsScaler, etc.).
                        Note: without a scaler, the loss is likely to explode.
        :type scaling: Optional[object]
        :param start_stamp: Optional index to start slicing the DataFrame for training/validation/testing.
        :type start_stamp: int
        :param end_stamp: Optional index to end slicing the DataFrame for training/validation/testing.
        :type end_stamp: Optional[int]
        :param gcp_service_key: Optional path to a GCP service key file (not currently used in implementation).
        :type gcp_service_key: Optional[str]
        :param interpolate_param: Flag or dictionary specifying interpolation parameters to handle NaNs.
        :type interpolate_param: Union[bool, Dict]
        :param sort_column: The column to sort the time series on prior to forecasting (typically a datetime column).
        :type sort_column: Optional[str]
        :param scaled_cols: The columns you want scaling applied to. If left blank, defaults to ``relevant_cols``.
        :type scaled_cols: Optional[List]
        :param feature_params: Parameters for generating temporal (datetime) features.
        :type feature_params: Optional[Dict]
        :param no_scale: If True, the target labels will not be scaled when returned by __getitem__.
        :type no_scale: bool
        :param preformatted_df: If True, assumes ``file_path`` is already a Pandas DataFrame (not currently used).
        :type preformatted_df: bool
        """
        super().__init__()
        interpolate = interpolate_param
        self.forecast_history = forecast_history
        self.forecast_length = forecast_length
        print("interpolate should be below")
        df = get_data(file_path)
        print(df.columns)
        relevant_cols3 = []
        if sort_column:
            df[sort_column] = to_tz_naive_datetime(df[sort_column])
            df = df.sort_values(by=sort_column)
            if feature_params:
                df, relevant_cols3 = feature_fix(feature_params, sort_column, df)
                print("Created datetime feature columns are: ")
        print(relevant_cols3)
        self.relevant_cols3 = relevant_cols3
        if interpolate:
            df = interpolate_dict[interpolate["method"]](df, **interpolate["params"])
        self.df = df[relevant_cols + relevant_cols3].copy()
        self.original_df = df
        self.scale = None
        if scaled_cols is None:
            scaled_cols = relevant_cols
        print("scaled cols are")
        print(scaled_cols)
        if start_stamp != 0 and end_stamp is not None:
            self.df = self.df[start_stamp:end_stamp]
        elif start_stamp != 0:
            self.df = self.df[start_stamp:]
        elif end_stamp is not None:
            self.df = self.df[:end_stamp]
        self.unscaled_df = self.df.copy()
        if scaling is not None:
            print("scaling now")
            self.scale = scaling.fit(self.df[scaled_cols])
            temp_df = self.scale.transform(self.df[scaled_cols])

            # We define a second scaler to scale the end output
            # back to normal as models might not necessarily predict
            # other present time series values.
            targ_scale_class = self.scale.__class__
            self.targ_scaler = targ_scale_class()
            self.df[target_col] = self.targ_scaler.fit_transform(self.df[target_col])

            self.df[scaled_cols] = temp_df
        if (len(self.df) - self.df.count()).max() != 0:
            print("Error nan values detected in data. Please run interpolate ffill or bfill on data")
        self.targ_col = target_col
        self.no_scale = no_scale

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample of historical data (src) and target data (trg) at a given index.

        :param idx: The starting index for the historical data slice.
        :type idx: int
        :return: A tuple containing the historical input data and the future target data.
                 (src_data, trg_dat)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        rows = self.df.iloc[idx: self.forecast_history + idx]
        targs_idx_start = self.forecast_history + idx
        if self.no_scale:
            targ_rows = self.unscaled_df.iloc[targs_idx_start: self.forecast_length + targs_idx_start]
        else:
            targ_rows = self.df.iloc[
                targs_idx_start: self.forecast_length + targs_idx_start
            ]
        src_data = rows.to_numpy()
        src_data = torch.from_numpy(src_data).float()
        trg_dat = targ_rows.to_numpy()
        trg_dat = torch.from_numpy(trg_dat).float()
        return src_data, trg_dat

    def __len__(self) -> int:
        """
        Returns the total number of possible samples (batches) that can be generated.
        The length accounts for the historical and forecast window sizes.

        :return: The number of available samples.
        :rtype: int
        """
        return (
            len(self.df.index) - self.forecast_history - self.forecast_length - 1
        )

    def __sample_and_track_series__(self, idx, series_id=None):
        """
        (Placeholder) Used for custom sampling logic in multi-series contexts.
        """
        pass

    def inverse_scale(
        self, result_data: Union[torch.Tensor, pd.Series, np.ndarray]
    ) -> torch.Tensor:
        """Un-does the scaling of the data using the target scaler (targ_scaler).

        :param result_data: The data you want to unscale (can handle multiple data types).
        :type result_data: Union[torch.Tensor, pd.Series, np.ndarray]
        :return: Returns the unscaled data as a PyTorch tensor.
        :rtype: torch.Tensor
        """
        if isinstance(result_data, pd.Series) or isinstance(
            result_data, pd.DataFrame
        ):
            result_data_np = result_data.values
        if isinstance(result_data, torch.Tensor):
            if len(result_data.shape) > 2:
                result_data = result_data.permute(2, 0, 1).reshape(result_data.shape[2], -1)
                result_data = result_data.permute(1, 0)
            elif len(result_data.shape) > 1:
                result_data = result_data
            else:
                result_data = result_data.unsqueeze(0)
            result_data_np = result_data.numpy()
        if isinstance(result_data, np.ndarray):
            result_data_np = result_data
        # print(type(result_data))
        if self.no_scale or self.scale is None:
            return torch.from_numpy(result_data_np)
        if len(result_data_np.shape) > 2:
            result_data_np = result_data_np[0, :, :]
        return torch.from_numpy(
            self.targ_scaler.inverse_transform(result_data_np)
        )


class CSVSeriesIDLoader(CSVDataLoader):
    """
    A data loader for a CSV file that contains multiple independent time series,
    distinguished by a series ID column. It returns data as dictionaries, keyed by series ID.
    """
    def __init__(self, series_id_col: str, main_params: dict, return_method: str, return_all=True):
        """Initializes the CSVSeriesIDLoader.

        :param series_id_col: The column name containing the unique series identifier.
        :type series_id_col: str
        :param main_params: The central set of parameters passed to the parent ``CSVDataLoader`` __init__.
        :type main_params: dict
        :param return_method: The method for returning data (e.g., 'dict').
        :type return_method: str
        :param return_all: Whether to return all series at once for each index, defaults to True.
                           If False, sampling logic (not implemented) would be used.
        :type return_all: bool, optional
        """
        main_params1 = deepcopy(main_params)
        if "scaled_cols" not in main_params1:
            main_params1["scaled_cols"] = main_params1["relevant_cols"].copy()
            print("The scaled cols are below")
            print(main_params1["scaled_cols"])
        main_params1["relevant_cols"].append(series_id_col)
        super().__init__(**main_params1)
        self.series_id_col = series_id_col
        self.return_method = return_method
        self.return_all_series = return_all
        self.unique_cols = self.original_df[series_id_col].dropna().unique().tolist()
        df_list = []
        self.df_orig_list = []
        self.df = self.df.reset_index()
        self.unique_dict = {}
        print("The series id column is below:")
        print(self.series_id_col)
        for col in self.unique_cols:
            self.df_orig_list.append(self.original_df[self.original_df[self.series_id_col] == col])
            new_df = self.df[self.df[self.series_id_col] == col]
            df_list.append(new_df)
            print(new_df.columns)
        self.listed_vals = df_list
        self.__make_unique_dict__()
        if return_all:
            self.__validate_data__in_df()
        print(self.unique_dict)
        print("unique dict")

    def __validate_data__in_df(self):
        """Checks if all sub-series DataFrames have equal length when ``return_all_series`` is True.

        :raises IndexError: If the length of sub-series data-frames are not equal.
        """
        if self.return_all_series:
            len_first = len(self.listed_vals[0])
            print("Length of first series is:" + str(len_first))
            for series in self.listed_vals:
                print("Length of first series is:" + str(len(series)))
                series_bool = len(series) == len_first
                if not series_bool:
                    raise IndexError("The length of sub-series data-frames are not equal.")

    def __make_unique_dict__(self):
        """Creates a mapping from unique series ID values to a sequential integer index."""
        for i in range(0, len(self.unique_cols)):
            self.unique_dict[self.unique_cols[i]] = i

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        """Returns a set of dictionaries that contain the historical (source) and target data for each series.

        :param idx: The index to lookup across all parallel series.
        :type idx: int
        :return: A tuple of dictionaries: (source_data_dict, target_data_dict).
                 Keys are sequential integer indices (from 0 to N-1), values are PyTorch Tensors.
        :rtype: Tuple[Dict, Dict]
        :raises NotImplementedError: If ``return_all_series`` is False.
        """
        if self.return_all_series:
            src_list = {}
            targ_list = {}
            for va in self.listed_vals:
                # We need to exclude the index column on one end and the series id column on the other

                targ_start_idx = idx + self.forecast_history
                idx2 = va[self.series_id_col].iloc[0]
                va_returned = va[va.columns.difference([self.series_id_col], sort=False)]
                t = torch.Tensor(va_returned.iloc[idx: self.forecast_history + idx].values)[:, 1:]
                targ = torch.Tensor(va_returned.iloc[targ_start_idx: targ_start_idx + self.forecast_length].to_numpy())[:, 1:]  # noqa
                src_list[self.unique_dict[idx2]] = t
                targ_list[self.unique_dict[idx2]] = targ
            return src_list, targ_list
        else:
            raise NotImplementedError
        return super().__getitem__(idx)

    def __sample_series_id__(idx, series_id):
        """
        (Placeholder) Used for sampling a single series from the multi-series dataset.
        """
        pass

    def __len__(self) -> int:
        """
        Returns the total number of possible samples (batches) that can be generated.

        :return: The number of available samples.
        :rtype: int
        :raises NotImplementedError: If ``return_all_series`` is False.
        """
        if self.return_all_series:
            return len(self.listed_vals[0]) - self.forecast_history - self.forecast_length - 1
        else:
            raise NotImplementedError("Current code only supports returning all the series at once at each iteration")


class CSVTestLoader(CSVDataLoader):
    """
    A data loader specifically for test data. It extends CSVDataLoader to return
    the original unscaled DataFrame slice along with the historical data tensor.
    """
    def __init__(
        self,
        df_path: str,
        forecast_total: int,
        use_real_precip=True,
        use_real_temp=True,
        target_supplied=True,
        interpolate=False,
        sort_column_clone=None,
        **kwargs
    ):
        """
        Initializes the CSVTestLoader.

        :param df_path: The path to the CSV file you want to use (GCS compatible) or a Pandas DataFrame.
        :type df_path: str
        :param forecast_total: The total length of the sequence that should be considered for forecasting
                               (history + forecast_length).
        :type forecast_total: int
        :param use_real_precip: (Antiquated/Deprecated) Flag for using real precipitation values.
        :type use_real_precip: bool
        :param use_real_temp: (Antiquated/Deprecated) Flag for using real temperature values.
        :type use_real_temp: bool
        :param target_supplied: Flag indicating if the target values are present in the test data.
        :type target_supplied: bool
        :param interpolate: Flag or dictionary specifying interpolation parameters to handle NaNs.
        :type interpolate: Union[bool, Dict]
        :param sort_column_clone: The column to sort the time series on prior to forecasting.
        :type sort_column_clone: Optional[str]
        :param kwargs: Additional keyword arguments passed to the parent ``CSVDataLoader``.
        """
        if "file_path" not in kwargs:
            kwargs["file_path"] = df_path
        super().__init__(**kwargs)
        df_path1 = df_path
        self.original_df = get_data(df_path1)
        if interpolate:
            self.original_df = interpolate_dict[interpolate["method"]](self.original_df, **interpolate["params"])
        if sort_column_clone:
            self.original_df = self.original_df.sort_values(by=sort_column_clone)
        print("CSV Path below")
        print(df_path)
        self.forecast_total = forecast_total
        # TODO these are antiquated delete them
        self.use_real_temp = use_real_temp
        self.use_real_precip = use_real_precip
        self.target_supplied = target_supplied
        # Convert back to datetime and save index
        sort_col1 = sort_column_clone if sort_column_clone else "datetime"
        print("columns are: ")
        print(self.original_df)
        self.original_df[sort_col1] = to_tz_naive_datetime(self.original_df["datetime"])
        self.original_df["original_index"] = self.original_df.index
        if len(self.relevant_cols3) > 0:
            self.original_df[self.relevant_cols3] = self.df[self.relevant_cols3]

    def get_from_start_date(self, forecast_start: datetime, original_df=None) -> Tuple[torch.Tensor, pd.DataFrame, int]:
        """
        Retrieves a sample starting from a specified datetime stamp.

        :param forecast_start: The datetime object indicating the start of the forecast window
                               (the first time step *after* the historical window).
        :type forecast_start: datetime
        :param original_df: Optional DataFrame to use instead of the internally stored one.
        :type original_df: Optional[pd.DataFrame]
        :return: The historical data, the original full sequence rows, and the target start index.
        :rtype: Tuple[torch.Tensor, pd.DataFrame, int]
        """
        if original_df is None:
            original_df = self.original_df
        dt_row = original_df[
            original_df["datetime"] == forecast_start
        ]
        revised_index = dt_row.index[0]
        return self.__getitem__(revised_index - self.forecast_history)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, pd.DataFrame, int]:
        """
        Retrieves a single test sample, consisting of historical data (scaled) and the full
        sequence of original data (unscaled).

        :param idx: The starting index for the historical data slice.
        :type idx: int
        :return: A tuple containing the historical input data tensor, the unscaled DataFrame slice
                 covering the whole sequence, and the index where the target sequence begins.
                 (historical_rows, all_rows_orig, target_idx_start)
        :rtype: Tuple[torch.Tensor, pd.DataFrame, int]
        """
        if self.target_supplied:
            historical_rows = self.df.iloc[idx: self.forecast_history + idx]
            target_idx_start = self.forecast_history + idx
            # Why aren't we using these
            # targ_rows = self.df.iloc[
            # target_idx_start : self.forecast_total + target_idx_start
            # ]
            all_rows_orig = self.original_df.iloc[
                idx: self.forecast_total + target_idx_start
            ].copy()
            historical_rows = torch.from_numpy(historical_rows.to_numpy())
            return historical_rows.float(), all_rows_orig, target_idx_start

    def convert_real_batches(self, the_col: str, rows_to_convert: pd.DataFrame) -> List[torch.Tensor]:
        """
        A helper function to return properly divided batches of data (e.g., precipitation or temperature)
        to be stacked with the forecasted target values.

        The data is chunked into lengths equal to ``self.forecast_length``.

        :param the_col: The name of the column to batch.
        :type the_col: str
        :param rows_to_convert: The DataFrame containing the column data.
        :type rows_to_convert: pd.DataFrame
        :return: A list of PyTorch tensors, where each tensor is a batch of size ``self.forecast_length``.
        :rtype: List[torch.Tensor]
        """
        the_column = torch.from_numpy(rows_to_convert[the_col].to_numpy())
        chunks = [
            the_column[
                self.forecast_length * i: self.forecast_length * (i + 1)
            ]
            for i in range(len(the_column) // self.forecast_length + 1)
        ]
        return chunks

    def convert_history_batches(
        self, the_col: Union[str, List[str]], rows_to_convert: pd.DataFrame
    ) -> List[torch.Tensor]:
        """A helper function to return dataframe in batches of size (history_len, num_features).

        The data is chunked into lengths equal to ``self.forecast_history``.

        :param the_col: Column name or list of column names.
        :type the_col: Union[str, List[str]]
        :param rows_to_convert: Rows in a DataFrame to be converted into batches.
        :type rows_to_convert: pd.DataFrame
        :return: A list of PyTorch tensors, where each tensor is a batch of size ``self.forecast_history``.
        :rtype: List[torch.Tensor]
        """
        the_column = torch.from_numpy(rows_to_convert[the_col].to_numpy())
        chunks = [
            the_column[
                self.forecast_history * i: self.forecast_history * (i + 1)
            ]
            for i in range(len(the_column) // self.forecast_history + 1)
        ]
        return chunks

    def __len__(self) -> int:
        """
        Returns the total number of possible test samples that can be generated.

        :return: The number of available test samples.
        :rtype: int
        """
        return (
            len(self.df.index) - self.forecast_history - self.forecast_total - 1
        )


class TestLoaderABC(CSVTestLoader):
    """
    (Abstract Base Class Placeholder) A placeholder class that inherits from CSVTestLoader.
    """
    pass


class AEDataloader(CSVDataLoader):
    """
    A data loader class tailored for **AutoEncoder (AE)** models.
    It overrides ``__len__`` and ``__getitem__`` from the generic ``CSVDataLoader``
    and defaults ``forecast_length`` to 1.
    """
    def __init__(
            self,
            file_path: str,
            relevant_cols: List,
            scaling=None,
            start_stamp: int = 0,
            target_col: List = None,
            end_stamp: int = None,
            unsqueeze_dim: int = 1,
            interpolate_param=False,
            forecast_history=1,
            no_scale=True,
            sort_column=None):
        """
        Initializes the AEDataloader.

        :param file_path: The path to the file.
        :type file_path: str
        :param relevant_cols: The relevant columns to be included in the input/output.
        :type relevant_cols: List
        :param scaling: Optional scaler object for data normalization, defaults to None.
        :type scaling: Optional[object]
        :param start_stamp: Optional index to start slicing the DataFrame, defaults to 0.
        :type start_stamp: int
        :param target_col: Optional list of target columns. For AE, this is usually the same as ``relevant_cols``.
        :type target_col: Optional[List]
        :param end_stamp: Optional index to end slicing the DataFrame, defaults to None.
        :type end_stamp: Optional[int]
        :param unsqueeze_dim: Dimension to unsqueeze the resulting tensor (not currently used in implementation).
        :type unsqueeze_dim: int
        :param interpolate_param: Flag or dictionary specifying interpolation parameters, defaults to False.
        :type interpolate_param: Union[bool, Dict]
        :param forecast_history: The sequence length for the autoencoder input, defaults to 1.
        :type forecast_history: int
        :param no_scale: If True, the target labels will not be scaled, defaults to True.
        :type no_scale: bool
        :param sort_column: The column to sort the time series on, defaults to None.
        :type sort_column: Optional[str]
        """
        super().__init__(file_path=file_path, forecast_history=forecast_history, forecast_length=1,
                         target_col=target_col, relevant_cols=relevant_cols, start_stamp=start_stamp,
                         end_stamp=end_stamp, sort_column=sort_column, interpolate_param=False, no_scale=no_scale,
                         scaling=scaling)
        self.unsqueeze_dim = unsqueeze_dim
        self.start_stamp = start_stamp

    def __handle_params__():
        """
        (Placeholder) For internal parameter handling logic.
        """
        pass

    def get_from_start_date(self, forecast_start: datetime) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample starting from a specified datetime stamp.

        :param forecast_start: The datetime object corresponding to the start of the sequence.
        :type forecast_start: datetime
        :return: A tuple containing the input data (src) and the target data (targ).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If the item was not found in the index.
        """
        dt_row = self.original_df[
            self.original_df["datetime"] == forecast_start
        ]
        revised_index = dt_row.index[0] - self.start_stamp
        return self.__getitem__(revised_index - self.forecast_history)

    def __len__(self) -> int:
        """
        Returns the total number of possible sequences that can be generated.

        :return: The number of available samples.
        :rtype: int
        """
        return len(self.df.index) - 1 - self.forecast_history

    def __getitem__(self, idx: int, uuid: int = None, column_relevant: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample for the autoencoder (src == target).

        :param idx: The starting index for the data slice.
        :type idx: int
        :param uuid: Optional UUID for lookups (not fully implemented).
        :type uuid: Optional[int]
        :param column_relevant: Optional column for UUID lookups (not fully implemented).
        :type column_relevant: Optional[str]
        :return: A tuple containing the input data and the target data (both are the same sequence).
                 (source_data, target_data)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If the item was not found in the index during UUID lookup.
        """
        # Warning this assumes that data is
        if uuid:
            idx = self.original_df[self.original_df[column_relevant] == uuid].index.values.astype(int)[0]
        target = torch.from_numpy(self.df.iloc[idx: idx + self.forecast_history].to_numpy()).float()
        if target.shape[0] == 0:
            raise ValueError("The item was not found in the index please try again")
        return torch.from_numpy(self.df.iloc[idx: idx + self.forecast_history].to_numpy()).float(), target


class GeneralClassificationLoader(CSVDataLoader):
    """
    A generic data loader class for time series classification problems.
    It returns a sequence of features (src) and a one-hot encoded classification label (targ).
    """
    def __init__(self, params: Dict, n_classes: int = 2):
        """Initializes the GeneralClassificationLoader.

        :param params: The standard dictionary for a dataloader, which must contain ``sequence_length``.
                       (See ``CSVDataLoader`` for other parameters).
        :type params: Dict
        :param n_classes: The number of classes in the classification problem, defaults to 2.
        :type n_classes: int
        """
        self.n_classes = n_classes
        params["forecast_history"] = params["sequence_length"]
        params["no_scale"] = True
        # This could really be anything as forecast_length is not used
        params["forecast_length"] = 1
        # Remove sequence_length prior to calling the super class
        params.pop("sequence_length")
        super().__init__(**params)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample of historical data (src) and a one-hot encoded class label (targ).

        The target is assumed to be the **first** column of the original data and is taken from the
        **last row** of the unscaled sequence.

        :param idx: The starting index for the sequence slice.
        :type idx: int
        :return: A tuple containing the feature sequence and the one-hot encoded label.
                 (feature_sequence, one_hot_label)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If the derived class value is greater than the specified number of classes.
        """
        rows = self.df.iloc[idx: self.forecast_history + idx]
        targ = self.unscaled_df.iloc[idx: self.forecast_history + idx]
        rows = torch.from_numpy(rows.to_numpy())
        targ = torch.from_numpy(targ.to_numpy())
        # Exclude the first row it is the target.
        src = rows[:, 1:]
        # Get label of the series sequence
        targ = targ[-1, 0]
        targ_labs = torch.zeros(self.n_classes)
        casted_shit = int(targ.data.tolist())
        if casted_shit > self.n_classes:
            raise ValueError("The class " + str(casted_shit) + " is greater than the number of classes " + str(self.n_classes))  # noqa
        targ_labs[casted_shit] = 1
        return src.float(), targ_labs.float().unsqueeze(0)


class TemporalLoader(CSVDataLoader):
    """
    A data loader class for creating and separating specific **temporal features** (e.g., year, month, day)
    from other time series features. This is often used for Informer-like models.
    """
    def __init__(
            self,
            time_feats: List[str],
            kwargs: Dict,
            label_len=0):
        """Initializes the TemporalLoader.

        :param time_feats: A list of strings of the temporal features to be separated (e.g., ['month', 'day', 'hour']).
        :type time_feats: List[str]
        :param kwargs: The set of parameters passed to the parent ``CSVDataLoader``.
        :type kwargs: Dict
        :param label_len: The label length used for Informer-based models, defaults to 0.
        :type label_len: int, optional
        """
        super().__init__(**kwargs)
        self.time_feats = time_feats
        self.temporal_df = self.df[time_feats]
        self.other_feats = self.df.drop(columns=time_feats)
        self.label_len = label_len

    @staticmethod
    def df_to_numpy(pandas_stuff: pd.DataFrame) -> torch.Tensor:
        """
        Converts a Pandas DataFrame into a float PyTorch Tensor.

        :param pandas_stuff: The DataFrame to convert.
        :type pandas_stuff: pd.DataFrame
        :return: The converted PyTorch tensor.
        :rtype: torch.Tensor
        """
        return torch.from_numpy(pandas_stuff.to_numpy()).float()

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieves a single sample, separating the main features and the temporal features for both source and target.

        :param idx: Index of the item to be returned.
        :type idx: int
        :return: A tuple containing the source and target data tuples:
                 ((src_data, temporal_feats), (tar_temp, trg_data))
        :rtype: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        """
        rows = self.other_feats.iloc[idx: self.forecast_history + idx]
        temporal_feats = self.temporal_df.iloc[idx: self.forecast_history + idx]
        targs_idx_start = self.forecast_history + idx - self.label_len
        targ_rows = self.other_feats.iloc[
            targs_idx_start: self.forecast_length + targs_idx_start + self.label_len
        ]
        targs_idx_s = targs_idx_start
        tar_temporal_feats = self.temporal_df.iloc[targs_idx_s: self.forecast_length + targs_idx_start + self.label_len]
        src_data = self.df_to_numpy(rows)
        trg_data = self.df_to_numpy(targ_rows)
        temporal_feats = self.df_to_numpy(temporal_feats)
        tar_temp = self.df_to_numpy(tar_temporal_feats)
        return (src_data, temporal_feats), (tar_temp, trg_data)

    def __len__(self) -> int:
        """
        Returns the total number of possible samples (batches) that can be generated.

        :return: The number of available samples.
        :rtype: int
        """
        return (
            len(self.df.index) - self.forecast_history - self.forecast_length - 1
        )


class TemporalTestLoader(CSVTestLoader):
    """
    A test data-loader class for test data in the format of the ``TemporalLoader``.
    It separates temporal features and main features for encoder and decoder inputs.
    """
    def __init__(self, time_feats: List[str], kwargs={}, decoder_step_len=None):
        """Initializes the TemporalTestLoader.

        :param time_feats: The temporal featuers to use in encoding.
        :type time_feats: List[str]
        :param kwargs: The dict used to instantiate ``CSVTestLoader`` parent (must contain ``df_path``
            and ``kwargs`` keys).
        :type kwargs: dict
        :param decoder_step_len: The length of the initial decoder input (label length for Informer), defaults to None.
        :type decoder_step_len: Optional[int]
        """
        super().__init__(kwargs["df_path"], kwargs["forecast_total"], **kwargs["kwargs"])
        self.time_feats = time_feats
        self.temporal_df = self.df[time_feats]
        self.other_feats = self.df.drop(columns=time_feats)
        self.decoder_step_len = decoder_step_len

    @staticmethod
    def df_to_numpy(pandas_stuff: pd.DataFrame) -> torch.Tensor:
        """
        Converts a Pandas DataFrame into a float PyTorch Tensor.

        :param pandas_stuff: The DataFrame to convert.
        :type pandas_stuff: pd.DataFrame
        :return: The converted PyTorch tensor.
        :rtype: torch.Tensor
        """
        return torch.from_numpy(pandas_stuff.to_numpy()).float()

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                             Tuple[torch.Tensor, torch.Tensor], pd.DataFrame, int]:
        """
        Retrieves a single test sample, separating features and returning the original unscaled data.

        :param idx: The starting index for the historical data slice.
        :type idx: int
        :return: A tuple containing:
                 1. Source data tuple: (historical main features, historical temporal features).
                 2. Target data tuple: (future temporal features for decoder, future main features (for target)).
                 3. The unscaled DataFrame slice covering the whole sequence.
                 4. The index where the target sequence begins.
        :rtype: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], pd.DataFrame, int]
        """
        if self.target_supplied:
            historical_rows = self.df.iloc[idx: self.forecast_history + idx]
            target_idx_start = self.forecast_history + idx
            # Why aren't we using these
            # targ_rows = self.df.iloc[
            # target_idx_start : self.forecast_total + target_idx_start
            # ]
            historical_rows = self.other_feats.iloc[idx: self.forecast_history + idx]
            targs_idx_start = self.forecast_history + idx
            temporal_feat = self.temporal_df.iloc[idx: self.forecast_history + idx]
            end_idx = self.forecast_total + target_idx_start
            if self.decoder_step_len:
                print("The label length is " + str(self.decoder_step_len))
                targs_idx_start = targs_idx_start - self.decoder_step_len
                print(targs_idx_start)
                target_idx_start = target_idx_start - self.decoder_step_len
                print(target_idx_start)
                end_idx = self.forecast_total + target_idx_start + self.decoder_step_len
                print(end_idx)
                tar_temporal_feats = self.temporal_df.iloc[targs_idx_start: end_idx]
                targ_rows = self.other_feats.iloc[targs_idx_start: end_idx]
            else:
                tar_temporal_feats = self.temporal_df.iloc[targs_idx_start: end_idx]
                targ_rows = self.other_feats.iloc[targs_idx_start: end_idx]
            src_data = self.df_to_numpy(historical_rows)
            trg_data = self.df_to_numpy(targ_rows)
            temporal_feat = self.df_to_numpy(temporal_feat)
            tar_temp = self.df_to_numpy(tar_temporal_feats)
            decoder_adjust = self.decoder_step_len if self.decoder_step_len else 0
            all_rows_orig = self.original_df.iloc[
                idx: self.forecast_total + target_idx_start + decoder_adjust
            ].copy()
            historical_rows = torch.from_numpy(historical_rows.to_numpy())
            return (src_data, temporal_feat), (tar_temp, trg_data), all_rows_orig, target_idx_start


class VariableSequenceLength(CSVDataLoader):
    """
    A data loader for time-series data where sequences (examples) have **variable length**.
    Sequences are grouped by a marker column and retrieved whole.
    """
    def __init__(self, series_marker_column: str, csv_loader_params: Dict, pad_length=None, task="classification",
                 n_classes=9 + 90):
        """Initializes the VariableSequenceLength loader.

        :param series_marker_column: The column that delineates when an example (sequence) begins and ends.
        :type series_marker_column: str
        :param csv_loader_params: The standard parameters passed to the parent ``CSVDataLoader``.
        :type csv_loader_params: Dict
        :param pad_length: If specified, the length to truncate sequences at or pad them up to.
        :type pad_length: Optional[int]
        :param task: The specific task ('classification', 'auto', 'forecasting' - not fully
            implemented), defaults to "classification".
        :type task: str
        :param n_classes: The maximum number of classes for classification tasks, defaults to 99.
        :type n_classes: int
        """
        super().__init__(**csv_loader_params)
        self.pad_length = pad_length
        self.series_marker_column = series_marker_column
        self.task = task
        self.uniques = self.df[series_marker_column].unique()
        self.grouped_df = self.df.groupby(series_marker_column)
        self.n_classes = n_classes

    def get_item_forecast(self, idx: int):
        """
        (Placeholder) Logic for sequence-to-sequence forecasting with variable length data.
        """
        pass

    def get_item_classification(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sequence for classification.

        The sequence's label is assumed to be the **first** column of the original data and is taken from the
        **last row** of the unscaled sequence.

        :param idx: The index of the unique series to retrieve (index in ``self.uniques``).
        :type idx: int
        :return: A tuple containing the feature sequence and the one-hot encoded label.
                 (feature_sequence, one_hot_label)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If the derived class value is greater than the specified number of classes.
        """
        item = self.grouped_df.get_group(self.uniques[idx])
        rows = item.iloc[idx: self.forecast_history + idx]
        targ = item.iloc[idx: self.forecast_history + idx]
        rows = torch.from_numpy(rows.to_numpy())
        targ = torch.from_numpy(targ.to_numpy())
        # Exclude the first row it is the target.
        src = rows[:, 1:]
        # Get label of the series sequence
        targ = targ[-1, 0]
        targ_labs = torch.zeros(self.n_classes)
        casted_shit = int(targ.data.tolist())
        if casted_shit > self.n_classes - 1:  # -1 because counting starts at zero
            raise ValueError("The class " + str(casted_shit) + " is greater than the number of classes " + str(self.n_classes))  # noqa
        targ_labs[casted_shit] = 1
        return src.float(), targ_labs.float().unsqueeze(0)

    def get_item_auto_encoder(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sequence for autoencoder training (src == target).
        Applies padding or truncation if ``pad_length`` is set.

        :param idx: The index of the unique series to retrieve (index in ``self.uniques``).
        :type idx: int
        :return: A tuple containing the padded/truncated sequence for both source and target.
                 (sequence, sequence)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        item = self.grouped_df.get_group(self.uniques[idx])
        the_seq = torch.from_numpy(item.to_numpy())
        if self.pad_length:
            res = self.pad_input_data(the_seq)
            return res.to(torch.float32), res.float()
        else:
            return the_seq.float(), the_seq.float()

    def pad_input_data(self, sequence: torch.Tensor) -> torch.Tensor:
        """Pads a sequence to a specified length or truncates it if longer.

        :param sequence: The input sequence tensor.
        :type sequence: torch.Tensor
        :return: The padded or truncated sequence.
        :rtype: torch.Tensor
        """
        if self.pad_length > sequence.shape[0]:
            pad_dim = self.pad_length - sequence.shape[0]
            return torch.nn.functional.pad(sequence, (0, 0, 0, pad_dim))
        else:
            return sequence[self.pad_length, :]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample based on the specified task (auto, classification).

        :param idx: The index of the unique series to retrieve.
        :type idx: int
        :return: A tuple containing the input data and the target data based on the task.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :raises KeyError: If the task is not defined in the tasks dictionary.
        """
        tasks = {"auto": self.get_item_auto_encoder, "classification": self.get_item_classification}
        return tasks[self.task](idx)


class SeriesIDTestLoader(CSVSeriesIDLoader):
    """
    A test data loader specifically for multi-series data, extending ``CSVSeriesIDLoader``
    to handle test-time sequence length requirements and to wrap each series in a
    ``CSVTestLoader``.
    """
    def __init__(self, series_id_col: str, main_params: dict, return_method: str, forecast_total=336, return_all=True):
        """Initializes the SeriesIDTestLoader.

        :param series_id_col: The column that contains the series_id.
        :type series_id_col: str
        :param main_params: The core parameters used to instantiate the parent ``CSVSeriesIDLoader``.
        :type main_params: dict
        :param return_method: The method of return (e.g., 'dict').
        :type return_method: str
        :param forecast_total: The total length to forecast, defaults to 336.
        :type forecast_total: int, optional
        :param return_all: Whether to return all series at once for each index, defaults to True.
        :type return_all: bool, optional
        """
        super().__init__(series_id_col, main_params, return_method, return_all)
        print("forecast_total is: " + str(forecast_total))
        self.forecast_total = forecast_total
        # NOTE: self.df_orig_list holds the original dataframes, which are passed to CSVTestLoader
        self.csv_test_loaders = [CSVTestLoader(loader_1, forecast_total, **main_params) for loader_1 in self.df_orig_list]  # noqa

    def get_from_start_date_all(self, forecast_start: datetime,
                                series_id: int = None) -> List[Tuple[torch.Tensor, pd.DataFrame, int]]:
        """
        Retrieves a sample for **all** series starting from a specified datetime stamp.

        :param forecast_start: The datetime object indicating the start of the forecast window.
        :type forecast_start: datetime
        :param series_id: Optional parameter for a specific series ID (not currently used for 'all' method).
        :type series_id: Optional[int]
        :return: A list of results, where each result is a tuple (historical_data, all_rows_orig, target_idx_start)
                 from the underlying ``CSVTestLoader`` for a single series.
        :rtype: List[Tuple[torch.Tensor, pd.DataFrame, int]]
        """
        res = []
        for test_loader in self.csv_test_loaders:
            res.append(test_loader.get_from_start_date(forecast_start))
        return res


class CatchmentWindowLoader(CSVDataLoader):
    """
    CSVDataLoader variant for (spin-up -> forecast-horizon) hydrology windows.

    Follows the parent's conventions (forecast_history = spin-up length, forecast_length = horizon)
    with three differences: the source window spans spin-up AND horizon rows (so state-space models
    can read observed history and forecast meteorology in one tensor) with the target columns zeroed
    in the horizon segment to prevent leakage; windows overlapping data gaps beyond a tolerance are
    skipped; and an optional drainage area converts a cfs target to mm/hr (the water-balance unit)
    before scaling. Physical channels (target flow, temperature, shortwave) should be excluded from
    scaling via the parent's ``scaled_cols`` so physics-based models receive real units.
    """

    def __init__(self, file_path, forecast_history: int, forecast_length: int, target_col: List,
                 relevant_cols: List, area_sq_km: float = None, min_valid_fraction: float = 0.95,
                 window_stride: int = 24, **kwargs):
        """
        Initializes the catchment window loader.

        :param file_path: CSV path (or DataFrame) with the gauge's hourly record.
        :type file_path: Union[str, pd.DataFrame]
        :param forecast_history: The spin-up window length in time steps.
        :type forecast_history: int
        :param forecast_length: The forecast horizon length in time steps.
        :type forecast_length: int
        :param target_col: The target column list, e.g. ["cfs"].
        :type target_col: List
        :param relevant_cols: Feature columns (including the target).
        :type relevant_cols: List
        :param area_sq_km: If given, converts the (cfs) target to mm/hr over this basin area,
            defaults to None.
        :type area_sq_km: float, optional
        :param min_valid_fraction: Minimum observed fraction of the combined window for it to be
            indexed, defaults to 0.95.
        :type min_valid_fraction: float, optional
        :param window_stride: Spacing between window start indices, defaults to 24.
        :type window_stride: int, optional
        :param kwargs: Remaining CSVDataLoader keyword arguments (scaling, sort_column, ...).
        :type kwargs: Dict
        """
        df = get_data(file_path)
        if area_sq_km is not None:
            for col in target_col:
                df[col] = df[col] * 0.0283168 * 3.6 / area_sq_km
        super().__init__(df, forecast_history, forecast_length, target_col, relevant_cols,
                         **kwargs)
        self.target_col_list = target_col
        observed = ~self.original_df[relevant_cols].isna().any(axis=1)
        observed = observed.to_numpy()
        window = forecast_history + forecast_length
        self.valid_starts = [start for start in range(0, len(self.df) - window, window_stride)
                             if observed[start:start + window].mean() >= min_valid_fraction]

    def __len__(self) -> int:
        """
        Returns the number of valid (gap-filtered) windows.

        :return: The window count.
        :rtype: int
        """
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        """
        Returns one (source, target) pair.

        :param idx: The valid-window index.
        :type idx: int
        :return: A tuple of (src of shape (forecast_history + forecast_length, n_features) with
            target columns zeroed in the horizon segment, trg of shape (forecast_length,
            n_features)) matching the parent's target convention.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        start = self.valid_starts[idx]
        split = start + self.forecast_history
        end = split + self.forecast_length
        src = self.df.iloc[start:end].copy()
        src.loc[src.index[self.forecast_history:], self.target_col_list] = 0.0
        trg = self.df.iloc[split:end]
        return (torch.from_numpy(src.to_numpy()).float(),
                torch.from_numpy(trg.to_numpy()).float())


class IdentityScaler:
    """
    A no-op stand-in for an sklearn scaler, for datasets whose tensors are already in the space
    the model trains in (e.g. :class:`MultiBasinWindowLoader`, which standardizes per basin).
    """

    def transform(self, values):
        """
        Returns the input unchanged as a NumPy array.

        :param values: The values to (not) transform.
        :type values: Union[np.ndarray, torch.Tensor, pd.DataFrame]
        :return: The same values as a NumPy array.
        :rtype: np.ndarray
        """
        return np.asarray(values)

    def inverse_transform(self, values):
        """
        Returns the input unchanged as a NumPy array.

        :param values: The values to (not) inverse-transform.
        :type values: Union[np.ndarray, torch.Tensor, pd.DataFrame]
        :return: The same values as a NumPy array.
        :rtype: np.ndarray
        """
        return np.asarray(values)


class MultiBasinWindowLoader(Dataset):
    """
    Combines per-basin :class:`CatchmentWindowLoader` instances into one training dataset.

    Driven by a basin *manifest* JSON (see ``experiments/catchment_foundation/build_manifest.py``)
    holding, per basin: the hourly CSV path, drainage area, a lapse-rate temperature offset,
    train-period met normalization stats and the train-period flow standard deviation in mm/hr
    (``flow_scale_mm_hr``). Each returned source window gains a trailing basin-index channel so a
    multi-basin model can look up per-basin context, and the target flow column is divided by the
    basin's ``flow_scale_mm_hr`` (per-basin flow standardization in the loss); the source flow stays
    physical for spin-up and assimilation. Exposes ``sample_weights`` (horizon-flow-variance window
    weights combined with a basin-frequency correction) and ``samples_per_epoch`` so the trainer can
    do variance-weighted sampling across windows and basins.
    """

    def __init__(self, manifest_path: str, forecast_history: int, forecast_length: int,
                 target_col: List, relevant_cols: List, scaled_cols: Optional[List] = None,
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 basin_split: Optional[str] = None, min_valid_fraction: float = 0.95,
                 window_stride: int = 24, samples_per_epoch: Optional[int] = None,
                 basin_sample_power: float = 0.5, datetime_col: str = "datetime",
                 max_basins: Optional[int] = None, require_hourly: bool = True,
                 min_target_coverage: float = 1.0, max_input_gap: int = 6):
        """
        Initializes the multi-basin window loader.

        :param manifest_path: Path to the basin manifest JSON (or an already-loaded dict).
        :type manifest_path: Union[str, Dict]
        :param forecast_history: The spin-up window length in time steps.
        :type forecast_history: int
        :param forecast_length: The forecast horizon length in time steps.
        :type forecast_length: int
        :param target_col: The target column list, e.g. ["cfs"].
        :type target_col: List
        :param relevant_cols: Feature columns (including the target and any derived columns the
            manifest's preprocessing section creates, e.g. a lapse-corrected temperature).
        :type relevant_cols: List
        :param scaled_cols: Columns standardized with the manifest's per-basin train-period stats;
            physical channels (target, raw temperature/shortwave) must be excluded, defaults to
            None which scales nothing.
        :type scaled_cols: List, optional
        :param start_date: Inclusive UTC date lower bound for this split, defaults to None.
        :type start_date: str, optional
        :param end_date: Exclusive UTC date upper bound for this split, defaults to None.
        :type end_date: str, optional
        :param basin_split: If given, only basins whose manifest ``split`` equals this value are
            loaded (e.g. "train" vs "holdout"), defaults to None which loads all.
        :type basin_split: str, optional
        :param min_valid_fraction: Minimum observed fraction for a window to be indexed,
            defaults to 0.95.
        :type min_valid_fraction: float, optional
        :param window_stride: Spacing between window start indices, defaults to 24.
        :type window_stride: int, optional
        :param samples_per_epoch: Number of weighted samples drawn per epoch by the trainer,
            defaults to None which uses the full window count.
        :type samples_per_epoch: int, optional
        :param basin_sample_power: Exponent on the per-basin window count when allocating sampling
            mass across basins (1.0 = proportional to record length, 0.0 = equal mass per basin),
            defaults to 0.5.
        :type basin_sample_power: float, optional
        :param datetime_col: Name of the timestamp column in the basin CSVs, defaults to
            "datetime".
        :type datetime_col: str, optional
        :param max_basins: Optional cap on the number of basins loaded (smoke runs),
            defaults to None.
        :type max_basins: int, optional
        :param require_hourly: Whether to reindex each basin onto a strict hourly grid so that
            absent rows become explicit gaps instead of silently compressing real time,
            defaults to True. Disable only to reproduce pre-fix behaviour.
        :type require_hourly: bool, optional
        :param min_target_coverage: Fraction of the horizon whose target flow must be genuinely
            observed for a window to be kept, defaults to 1.0 (never score against interpolated
            flow). Values below 1.0 admit windows whose targets are partly imputed.
        :type min_target_coverage: float, optional
        :param max_input_gap: Longest run of missing steps that may be interpolated in the INPUT
            channels, in hours, defaults to 6. Longer runs are left missing, which rejects the
            windows containing them.
        :type max_input_gap: int, optional
        """
        import json
        super().__init__()
        manifest = manifest_path
        if not isinstance(manifest, dict):
            with open(manifest_path) as f:
                manifest = json.load(f)
        self.manifest = manifest
        self.forecast_history = forecast_history
        self.forecast_length = forecast_length
        self.target_col_list = target_col
        self.relevant_cols = relevant_cols
        self.require_hourly = require_hourly
        self.min_target_coverage = min_target_coverage
        self.max_input_gap = max_input_gap
        self.no_scale = True
        self.scale = None
        self.targ_scaler = IdentityScaler()
        selected = [(pos, b) for pos, b in enumerate(manifest["basins"])
                    if basin_split is None or b.get("split") == basin_split]
        if max_basins is not None:
            selected = selected[:max_basins]
        prep = manifest.get("preprocessing", {})
        self.basin_positions: List[int] = []
        self.basin_site_ids: List[str] = []
        self.basin_loaders: List[CatchmentWindowLoader] = []
        self.basin_timestamps: List[pd.DatetimeIndex] = []
        self.flow_scales: List[float] = []
        weight_blocks: List[np.ndarray] = []
        for pos, basin in selected:
            loader, timestamps = self._build_basin_loader(
                basin, prep, scaled_cols, start_date, end_date, min_valid_fraction,
                window_stride, datetime_col)
            if loader is None or len(loader) == 0:
                continue
            self.basin_positions.append(pos)
            self.basin_site_ids.append(basin["site_id"])
            self.basin_loaders.append(loader)
            self.basin_timestamps.append(timestamps)
            self.flow_scales.append(float(basin["flow_scale_mm_hr"]))
            weight_blocks.append(self._window_weights(loader, basin_sample_power))
        if not self.basin_loaders:
            raise ValueError("No basins with valid windows for split=%s in [%s, %s)"
                             % (basin_split, start_date, end_date))
        self.cumulative_windows = np.cumsum([len(loader) for loader in self.basin_loaders])
        self.sample_weights = torch.from_numpy(np.concatenate(weight_blocks)).double()
        self.samples_per_epoch = samples_per_epoch

    def _build_basin_loader(self, basin: Dict, prep: Dict, scaled_cols: Optional[List],
                            start_date: Optional[str], end_date: Optional[str],
                            min_valid_fraction: float, window_stride: int, datetime_col: str):
        """
        Reads, derives, slices and standardizes one basin's record and wraps it in a
        :class:`CatchmentWindowLoader`.

        :param basin: The basin's manifest entry.
        :type basin: Dict
        :param prep: The manifest's preprocessing section (fill_from / copy_cols / lapse).
        :type prep: Dict
        :param scaled_cols: Columns standardized with the basin's ``met_stats``.
        :type scaled_cols: List, optional
        :param start_date: Inclusive split lower bound.
        :type start_date: str, optional
        :param end_date: Exclusive split upper bound.
        :type end_date: str, optional
        :param min_valid_fraction: Minimum observed fraction per window.
        :type min_valid_fraction: float
        :param window_stride: Spacing between window starts.
        :type window_stride: int
        :param datetime_col: The timestamp column name.
        :type datetime_col: str
        :return: A (loader, timestamps) tuple; (None, None) when the slice is empty.
        :rtype: Tuple[Optional[CatchmentWindowLoader], Optional[pd.DatetimeIndex]]
        """
        derived = set(prep.get("copy_cols", {}))
        derived.add(prep.get("lapse", {}).get("target"))
        derived.add(prep.get("swe_col"))
        derived.update(prep.get("observed_mask_cols", {}))
        base_cols = [col for col in self.relevant_cols if col not in derived]
        sources = list(prep.get("fill_from", {}).values()) + \
            list(prep.get("copy_cols", {}).values()) + \
            list(prep.get("observed_mask_cols", {}).values())
        if prep.get("lapse"):
            sources.append(prep["lapse"]["source"])
        extra = sorted(set(source for source in sources if source not in base_cols))
        header = pd.read_csv(basin["csv_path"], nrows=0).columns
        wanted = [col for col in base_cols + extra
                  if col in header or col not in prep.get("fill_from", {})]
        frame = pd.read_csv(basin["csv_path"], usecols=[datetime_col] + wanted)
        frame[datetime_col] = to_tz_naive_datetime(frame[datetime_col])
        frame = frame.sort_values(datetime_col).drop_duplicates(datetime_col)
        frame = frame.set_index(datetime_col)
        if self.require_hourly:
            # Reindex onto a strict hourly grid. Without this, windows are sliced by ROW COUNT, so
            # absent rows silently compress real time -- nominal 1,056-row windows have been
            # observed spanning 22,237 real hours, and the ODE integrates a multi-hour gap as one
            # hour. Reindexing makes absent rows explicit NaNs that the validity filter can see.
            frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="h"))
        # Provenance masks MUST be taken before fill_from runs, otherwise a station column that
        # was silently backfilled from the gridded product is indistinguishable from one where the
        # station genuinely agreed with the grid.
        for mask_col, source in prep.get("observed_mask_cols", {}).items():
            if mask_col in self.relevant_cols:
                present = frame[source].notna() if source in frame.columns else False
                frame[mask_col] = np.asarray(present, dtype=np.float32)
        for col, source in prep.get("fill_from", {}).items():
            if col not in frame.columns:
                frame[col] = frame[source]
            else:
                frame[col] = frame[col].fillna(frame[source])
        for col, source in prep.get("copy_cols", {}).items():
            frame[col] = frame[source]
        lapse = prep.get("lapse")
        if lapse:
            frame[lapse["target"]] = frame[lapse["source"]] + basin.get("temp_offset_c", 0.0)
        swe_col = prep.get("swe_col")
        if swe_col and swe_col in self.relevant_cols:
            frame[swe_col] = self._swe_column(basin.get("swe_csv_path"), frame.index)
        if start_date is not None:
            frame = frame[frame.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            frame = frame[frame.index < pd.Timestamp(end_date)]
        if len(frame) <= self.forecast_history + self.forecast_length:
            return None, None
        frame = frame[self.relevant_cols].astype(np.float32)
        stats = basin.get("met_stats", {})
        for col in scaled_cols or []:
            mean, std = stats.get(col, (0.0, 1.0))
            frame[col] = (frame[col] - mean) / max(std, 1e-8)
        timestamps = frame.index
        target_observed = frame[self.target_col_list[0]].notna().to_numpy()
        loader = CatchmentWindowLoader(frame.reset_index(drop=True), self.forecast_history,
                                       self.forecast_length, self.target_col_list,
                                       self.relevant_cols, area_sq_km=basin["area_sq_km"],
                                       min_valid_fraction=min_valid_fraction,
                                       window_stride=window_stride, no_scale=True)
        loader.valid_starts = self._valid_windows(loader.valid_starts, target_observed)
        # Fill only the INPUT holes the validity filter tolerated, and only across short runs --
        # never bidirectionally across arbitrary spans. Target flow is never fabricated: windows
        # whose target is not fully observed have already been rejected above.
        loader.df = loader.df.interpolate(limit=self.max_input_gap,
                                          limit_direction="both").astype(np.float32)
        # Bounded interpolation deliberately leaves long gaps unfilled, so a window can still hold
        # NaNs in an INPUT column even when its target is fully observed. Drop those outright:
        # every served window must be finite, or the NaN reaches the ODE and the run dies.
        finite = np.isfinite(loader.df.to_numpy()).all(axis=1)
        cumulative = np.concatenate([[0], np.cumsum(finite)])
        span = self.forecast_history + self.forecast_length
        loader.valid_starts = [start for start in loader.valid_starts
                               if cumulative[start + span] - cumulative[start] == span]
        # Drop init-time copies that __getitem__ never touches (memory: O(100) basins).
        loader.original_df = None
        loader.unscaled_df = None
        return loader, timestamps

    def _valid_windows(self, starts: List[int], target_observed: np.ndarray) -> List[int]:
        """
        Keeps only windows whose flow observations support an honest forecast and score.

        Two rules, both rejections rather than imputations: the flow at issue time must be a
        genuine observation (the whole forecast is conditioned on it through
        ``match_current_flow``, so assimilating an interpolated value corrupts everything
        downstream), and the horizon target must be observed at least ``min_target_coverage`` of
        the time. Interpolating a target would mean scoring the model partly against our own
        interpolation, which is smooth by construction and therefore biased in favour of the
        smooth-drift failure mode we are trying to detect.

        :param starts: Candidate window start indices.
        :type starts: List[int]
        :param target_observed: Boolean array, True where the target column is observed.
        :type target_observed: np.ndarray
        :return: The surviving window start indices.
        :rtype: List[int]
        """
        spinup, horizon = self.forecast_history, self.forecast_length
        kept = []
        for start in starts:
            if not target_observed[start + spinup - 1]:
                continue  # issue-time flow must be real
            segment = target_observed[start + spinup:start + spinup + horizon]
            if segment.mean() >= self.min_target_coverage:
                kept.append(start)
        return kept

    def _swe_column(self, swe_csv_path: Optional[str],
                    index: pd.DatetimeIndex) -> np.ndarray:
        """
        Builds the hourly observed-SWE channel from a daily basin-mean series.

        Each daily value (e.g. a SNODAS basin mean) is forward-filled across its calendar day;
        hours without an observation get the -1.0 sentinel, which the model treats as "no
        observation" (falling back to its default empty-snow-store initialization). Physically
        missing == zero SWE only outside the snow season, which is exactly when the scraper
        skips days, so the sentinel is safe there too.

        :param swe_csv_path: Path of a CSV with "datetime" (daily) and "snodas_swe_mm" columns;
            None when the basin has no series, which yields an all-sentinel channel.
        :type swe_csv_path: str, optional
        :param index: The basin frame's hourly DatetimeIndex.
        :type index: pd.DatetimeIndex
        :return: The SWE channel of shape (len(index),) in mm with -1.0 sentinels.
        :rtype: np.ndarray
        """
        if not swe_csv_path or not os.path.exists(swe_csv_path):
            return np.full(len(index), -1.0, dtype=np.float32)
        daily = pd.read_csv(swe_csv_path)
        series = pd.Series(daily["snodas_swe_mm"].to_numpy(),
                           index=pd.to_datetime(daily["datetime"]))
        values = series.reindex(index.floor("D")).to_numpy(dtype=np.float32)
        return np.where(np.isfinite(values), values, -1.0).astype(np.float32)

    def _window_weights(self, loader: CatchmentWindowLoader,
                        basin_sample_power: float) -> np.ndarray:
        """
        Computes sampling weights for one basin's windows.

        Within the basin, windows are weighted by their horizon flow variance (floored at 0.1 of
        the basin mean so recessions still appear); across basins, total mass is allocated
        proportionally to ``n_windows ** basin_sample_power`` so long records do not drown out
        short ones.

        :param loader: The basin's window loader (post-interpolation).
        :type loader: CatchmentWindowLoader
        :param basin_sample_power: Exponent on the basin's window count for its total mass.
        :type basin_sample_power: float
        :return: Weights of shape (len(loader),).
        :rtype: np.ndarray
        """
        flow = loader.df[self.target_col_list[0]].to_numpy()
        history = self.forecast_history
        variances = np.array([np.nanvar(flow[s + history:s + history + self.forecast_length])
                              for s in loader.valid_starts])
        weights = np.maximum(variances / max(variances.mean(), 1e-12), 0.1)
        mass = len(loader) ** basin_sample_power
        return weights * (mass / weights.sum())

    def __len__(self) -> int:
        """
        Returns the total window count across basins.

        :return: The number of windows.
        :rtype: int
        """
        return int(self.cumulative_windows[-1])

    def locate(self, idx: int) -> Tuple[int, int]:
        """
        Maps a global window index to (basin position in this dataset, local window index).

        :param idx: The global window index.
        :type idx: int
        :return: A (basin_index, local_index) tuple.
        :rtype: Tuple[int, int]
        """
        basin = int(np.searchsorted(self.cumulative_windows, idx, side="right"))
        prior = 0 if basin == 0 else int(self.cumulative_windows[basin - 1])
        return basin, idx - prior

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one (source, target) pair with the basin-index channel appended to the source and
        the target flow standardized by the basin's flow scale.

        :param idx: The global window index.
        :type idx: int
        :return: A tuple of (src of shape (spin-up + horizon, n_features + 1),
            trg of shape (horizon, n_features + 1)); src and trg share the trailing basin
            channel so sequence-decoding utilities can concatenate them.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        basin, local = self.locate(idx)
        src, trg = self.basin_loaders[basin][local]
        trg = trg.clone()
        trg[:, 0] = trg[:, 0] / self.flow_scales[basin]
        position = float(self.basin_positions[basin])
        src_marker = torch.full((src.shape[0], 1), position)
        trg_marker = torch.full((trg.shape[0], 1), position)
        return (torch.cat([src, src_marker], dim=1), torch.cat([trg, trg_marker], dim=1))
