from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Union, Optional, List
from flood_forecast.pre_dict import interpolate_dict
from flood_forecast.preprocessing.buil_dataset import get_data
from datetime import datetime
from flood_forecast.preprocessing.temporal_feats import feature_fix
from copy import deepcopy


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
            df[sort_column] = df[sort_column].astype("datetime64[ns]")
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
        self.df.to_csv("temp_df.csv")
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
        self.original_df[sort_col1] = self.original_df["datetime"].astype("datetime64[ns]")
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
        :param kwargs: The dict used to instantiate ``CSVTestLoader`` parent (must contain ``df_path`` and ``kwargs`` keys).
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

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], pd.DataFrame, int]:
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
        :param task: The specific task ('classification', 'auto', 'forecasting' - not fully implemented), defaults to "classification".
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

    def get_from_start_date_all(self, forecast_start: datetime, series_id: int = None) -> List[Tuple[torch.Tensor, pd.DataFrame, int]]:
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
