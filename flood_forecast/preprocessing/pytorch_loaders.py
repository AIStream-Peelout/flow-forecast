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
import logging


class CSVDataLoader(Dataset):
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
        A data loader that takes a CSV file and properly batches for use in training/eval a PyTorch model

        :param file_path: The path to the CSV file you wish to use (GCS compatible) or a Pandas dataframe.
        :param forecast_history: This is the length of the historical time series data you wish to
                                utilize for forecasting
        :param forecast_length: The number of time steps to forecast ahead (for transformer this must
                                equal history_length)
        :param relevant_cols: Supply column names you wish to predict in the forecast (others will not be used)
        :param target_col: The target column or columns you to predict. If you only have one still use a list ['cfs']
        :param scaling: (highly recommended) If provided should be a subclass of sklearn.base.BaseEstimator
        and sklearn.base.TransformerMixin) i.e StandardScaler,  MaxAbsScaler, MinMaxScaler, etc) Note without
        a scaler the loss is likely to explode and cause infinite loss which will corrupt weights
        :param start_stamp int: Optional if you want to only use part of a CSV for training, validation
                                or testing supply these
        :param end_stamp int: Optional if you want to only use part of a CSV for training, validation,
                            or testing supply these
        :param sort_column str: The column to sort the time series on prior to forecast.
        :param scaled_cols: The columns you want scaling applied to (if left blank will default to all columns)
        :param feature_params: These are the datetime features you want to create.
        :param no_scale: This means that the end labels will not be scaled when running
        """
        super().__init__()
        interpolate = interpolate_param
        self.forecast_history = forecast_history
        self.forecast_length = forecast_length
        logging.log(logging.INFO, "Now loading the CSV file from: " + file_path)
        df = get_data(file_path)
        logging.log(logging.INFO, "Found the following columns in the CSV file: " + str(df.columns))
        relevant_cols3 = []
        if sort_column:
            df[sort_column] = df[sort_column].astype("datetime64[ns]")
            df = df.sort_values(by=sort_column)
            if feature_params:
                df, relevant_cols3 = feature_fix(feature_params, sort_column, df)
                logging.log(logging.INFO, "Created the following columns: " + str(relevant_cols3))
        self.relevant_cols3 = relevant_cols3
        if interpolate:
            df = interpolate_dict[interpolate["method"]](df, **interpolate["params"])
        self.df = df[relevant_cols + relevant_cols3].copy()
        self.original_df = df
        self.scale = None
        if scaled_cols is None:
            scaled_cols = relevant_cols
        # logging.log(logging.INFO, "The scaled columns are: " + str(scaled_cols))
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

    def __getitem__(self, idx: int):
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
        return (
            len(self.df.index) - self.forecast_history - self.forecast_length - 1
        )

    def __sample_and_track_series__(self, idx, series_id=None):
        pass

    def inverse_scale(
        self, result_data: Union[torch.Tensor, pd.Series, np.ndarray]
    ) -> torch.Tensor:
        """Un-does the scaling of the data

        :param result_data: The data you want to unscale can handle multiple data types.
        :type result_data: Union[torch.Tensor, pd.Series, np.ndarray]
        :return: Returns the unscaled data as PyTorch tensor.
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
    def __init__(self, series_id_col: str, main_params: dict, return_method: str, return_all=True):
        """A data-loader for a CSV file that contains a series ID column.

        :param series_id_col: The id
        :type series_id_col: str
        :param main_params: The central set of parameters
        :type main_params: dict
        :param return_method: The method of return
        :type return_method: str
        :param return_all: Whether to return all items, defaults to True
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
        """Makes sure the data in the data-frame is the proper length for each series e
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
        for i in range(0, len(self.unique_cols)):
            self.unique_dict[self.unique_cols[i]] = i

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        """Returns a set of dictionaries that contain the data for each series.

        :param idx: The index to lookup in the dataframe
        :type idx: int
        :return: A set of dictionaries that contain the data for each series.
        :rtype: Tuple[Dict, Dict]
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
        pass

    def __len__(self) -> int:
        if self.return_all_series:
            return len(self.listed_vals[0]) - self.forecast_history - self.forecast_length - 1
        else:
            raise NotImplementedError("Current code only supports returning all the series at once at each iteration")


class CSVTestLoader(CSVDataLoader):
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
        :param str df_path: The path to the CSV file you want to use (GCS compatible) or a Pandas DataFrame
        A data loader for the test data.
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

    def get_from_start_date(self, forecast_start: datetime, original_df=None):
        if original_df is None:
            original_df = self.original_df
        dt_row = original_df[
            original_df["datetime"] == forecast_start
        ]
        revised_index = dt_row.index[0]
        return self.__getitem__(revised_index - self.forecast_history)

    def __getitem__(self, idx):
        if self.target_supplied:
            historical_rows = self.df.iloc[idx: self.forecast_history + idx]
            target_idx_start = self.forecast_history + idx
            # Why aren't we using these
            # targ_rows = self.df.iloc[
            #     target_idx_start : self.forecast_total + target_idx_start
            # ]
            all_rows_orig = self.original_df.iloc[
                idx: self.forecast_total + target_idx_start
            ].copy()
            historical_rows = torch.from_numpy(historical_rows.to_numpy())
            return historical_rows.float(), all_rows_orig, target_idx_start

    def convert_real_batches(self, the_col: str, rows_to_convert):
        """
        A helper function to return properly divided precip and temp
        values to be stacked with t forecasted cfs.
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
    ):
        """A helper function to return dataframe in batches of
        size (history_len, num_features)

        Args:
            the_col (str): column names
            rows_to_convert (pd.Dataframe): rows in a dataframe
            to be converted into batches
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
        return (
            len(self.df.index) - self.forecast_history - self.forecast_total - 1
        )


class TestLoaderABC(CSVTestLoader):
    pass


class AEDataloader(CSVDataLoader):
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
        """A data loader class for autoencoders. Overrides __len__ and __getitem__ from generic dataloader.
           Also defaults forecast_history and forecast_length to 1. Since AE will likely only use one row.
           Same parameters as before.

        :param file_path: The path to the file
        :type file_path: str
        :param relevant_cols: The relevant columns
        :type relevant_cols: List
        :param scaling: [description], defaults to None
        :type scaling: [type], optional
        :param start_stamp: [description], defaults to 0
        :type start_stamp: int, optional
        :param target_col: [description], defaults to None
        :type target_col: List, optional
        :param end_stamp: [description], defaults to None
        :type end_stamp: int, optional
        :param unsqueeze_dim: [description], defaults to 1
        :type unsqueeze_dim: int, optional
        :param interpolate_param: [description], defaults to False
        :type interpolate_param: bool, optional
        :param forecast_history: [description], defaults to 1
        :type forecast_history: int, optional
        :param no_scale: [description], defaults to True
        :type no_scale: bool, optionals
        :param sort_column: [description], defaults to None
        :type sort_column: [type], optional
        """
        super().__init__(file_path=file_path, forecast_history=forecast_history, forecast_length=1,
                         target_col=target_col, relevant_cols=relevant_cols, start_stamp=start_stamp,
                         end_stamp=end_stamp, sort_column=sort_column, interpolate_param=False, no_scale=no_scale,
                         scaling=scaling)
        self.unsqueeze_dim = unsqueeze_dim
        self.start_stamp = start_stamp

    def __handle_params__():
        pass

    def get_from_start_date(self, forecast_start: datetime):
        dt_row = self.original_df[
            self.original_df["datetime"] == forecast_start
        ]
        revised_index = dt_row.index[0] - self.start_stamp
        return self.__getitem__(revised_index - self.forecast_history)

    def __len__(self):
        return len(self.df.index) - 1 - self.forecast_history

    def __getitem__(self, idx: int, uuid: int = None, column_relevant: str = None):
        # Warning this assumes that data is
        if uuid:
            idx = self.original_df[self.original_df[column_relevant] == uuid].index.values.astype(int)[0]
        target = torch.from_numpy(self.df.iloc[idx: idx + self.forecast_history].to_numpy()).float()
        if target.shape[0] == 0:
            raise ValueError("The item was not found in the index please try again")
        return torch.from_numpy(self.df.iloc[idx: idx + self.forecast_history].to_numpy()).float(), target


class GeneralClassificationLoader(CSVDataLoader):
    def __init__(self, params: Dict, n_classes: int = 2):
        """A generic data loader class for TS classification problems.

        :param params: The standard dictionary for a dataloader (see CSVDataLoader)
        :type params: Dict
        :param n_classes: The number of classes in the problem
        """  # noqa
        self.n_classes = n_classes
        params["forecast_history"] = params["sequence_length"]
        params["no_scale"] = True
        # This could really be anything as forecast_length is not used
        params["forecast_length"] = 1
        # Remove sequence_length prior to calling the super class
        params.pop("sequence_length")
        super().__init__(**params)

    def __getitem__(self, idx: int):
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
    def __init__(
            self,
            time_feats: List[str],
            kwargs: Dict,
            label_len=0):
        """A data loader class for creating specific temporal features/embeddings.

        :param time_feats: A list of strings of the time features (e.g. ['month', 'day', 'hour'])
        :type time_feats: List[str]
        :param kwargs: The set of parameters
        :type kwargs: Dict[str, Any]
        :param label_len: For Informer based model the, defaults to 0
        :type label_len: int, optional
        """
        super().__init__(**kwargs)
        self.time_feats = time_feats
        self.temporal_df = self.df[time_feats]
        self.other_feats = self.df.drop(columns=time_feats)
        self.label_len = label_len

    @staticmethod
    def df_to_numpy(pandas_stuff: pd.DataFrame):
        return torch.from_numpy(pandas_stuff.to_numpy()).float()

    def __getitem__(self, idx: int):
        """
        :param idx: Index of the item to be returned
        .. highlight:: python
        .. code-block:: python
            ## Example data
            ## -----------------
            ## 1992-01-01    0.0
            ## 1992-01-02    1.0
            ## 1992-01-03    2.0
            ## 1992-01-04    3.0
            ## 1992-01-05    4.0
            ## 1992-01-06    5.0
            ## -----------------
            kwargs = {"forecast_history" : 4, "forecast_length" : 2, "batch_size" : 1, "shuffle" : False,
            "num_workers" : 1}
            d = TemporalLoader(time_feats=["year", "month"], kwargs, label_len=1)
            x, y = d[0]
            print(x[0]) # (tensor([[0.0, 1.0, 2.0, 3.0]]))]),
            print(y[0]) # (tensor([[3.0, 4.0, 5.0, 6.0]]))])
            print(x[1]) # ,

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
        return (
            len(self.df.index) - self.forecast_history - self.forecast_length - 1
        )


class TemporalTestLoader(CSVTestLoader):
    def __init__(self, time_feats: List[str], kwargs={}, decoder_step_len=None):
        """A test data-loader class for data in the format of the TemporalLoader.

        :param time_feats: The temporal featuers to use in encoding.
        :type time_feats: List[str]
        :param kwargs: The dict used to instantiate CSVTestLoader parent, defaults to {}
        :type kwargs: dict, optional
        :param decoder_step_len: [description], defaults to None
        :type decoder_step_len: [type], optional

        ...
        ...
        """
        super().__init__(kwargs["df_path"], kwargs["forecast_total"], **kwargs["kwargs"])
        self.time_feats = time_feats
        self.temporal_df = self.df[time_feats]
        self.other_feats = self.df.drop(columns=time_feats)
        self.decoder_step_len = decoder_step_len

    @staticmethod
    def df_to_numpy(pandas_stuff: pd.DataFrame):
        return torch.from_numpy(pandas_stuff.to_numpy()).float()

    def __getitem__(self, idx):
        if self.target_supplied:
            historical_rows = self.df.iloc[idx: self.forecast_history + idx]
            target_idx_start = self.forecast_history + idx
            # Why aren't we using these
            # targ_rows = self.df.iloc[
            #     target_idx_start : self.forecast_total + target_idx_start
            historical_rows = self.other_feats.iloc[idx: self.forecast_history + idx]
            targs_idx_start = self.forecast_history + idx
            temporal_feat = self.temporal_df.iloc[idx: self.forecast_history + idx]
            end_idx = self.forecast_total + target_idx_start
            if self.decoder_step_len:
                print("The label length is " + str(self.decoder_step_len))
                targs_idx_start = targs_idx_start - self.decoder_step_len
                print(targs_idx_start)
                target_idx_start = target_idx_start - self.decoder_step_len
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
    def __init__(self, series_marker_column: str, csv_loader_params: Dict, pad_length=None, task="classification",
                 n_classes=9 + 90):
        """Enables eas(ier) loading of time-series with variable length data

        :param series_marker_column: The column that dealinates when an example begins and ends
        :type series_marker_column: str
        :param pad_length: If the specified the length to truncate sequences at or pad them till that length
        :type pad_length: int
        :param task: The specific task (e.g. classification, forecasting, auto_encode)
        :type task: str

        """
        super().__init__(**csv_loader_params)
        self.pad_length = pad_length
        self.series_marker_column = series_marker_column
        self.task = task
        self.uniques = self.df[series_marker_column].unique()
        self.grouped_df = self.df.groupby(series_marker_column)
        self.n_classes = n_classes

    def get_item_forecast(self, idx: int):
        pass

    def get_item_classification(self, idx: int):
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

    def get_item_auto_encoder(self, idx):
        item = self.grouped_df.get_group(self.uniques[idx])
        the_seq = torch.from_numpy(item.to_numpy())
        if self.pad_length:
            res = self.pad_input_data(the_seq)
            return res.to(torch.float32), res.float()
        else:
            return the_seq.float(), the_seq.float()

    def pad_input_data(self, sequence: int):
        """Pads a sequence to a specified length.
        """
        if self.pad_length > sequence.shape[0]:
            pad_dim = self.pad_length - sequence.shape[0]
            return torch.nn.functional.pad(sequence, (0, 0, 0, pad_dim))
        else:
            return sequence[self.pad_length, :]

    def __getitem__(self, idx: int):
        tasks = {"auto": self.get_item_auto_encoder, "classification": self.get_item_classification}
        return tasks[self.task](idx)


class SeriesIDTestLoader(CSVSeriesIDLoader):
    def __init__(self, series_id_col: str, main_params: dict, return_method: str, forecast_total=336, return_all=True):
        """A test loader for generating

        :param series_id_col: The column that contains the series_id
        :type series_id_col: str
        :param main_params: The core params used to instantiate the CSVSeriesIDLoader
        :type main_params: dict
        :param return_method: The method of return
        :type return_method: str
        :param return_all: _description_, defaults to True
        :type return_all: bool, optional
        :param forecast_total: The total length to forecast, defaults to 336
        :type forecast_total: int, optional
        """
        super().__init__(series_id_col, main_params, return_method, return_all)
        print("forecast_total is: " + str(forecast_total))
        self.forecast_total = forecast_total
        self.csv_test_loaders = [CSVTestLoader(loader_1, forecast_total, **main_params) for loader_1 in self.df_orig_list]  # noqa

    def get_from_start_date_all(self, forecast_start: datetime, series_id: int = None):
        res = []
        for test_loader in self.csv_test_loaders:
            res.append(test_loader.get_from_start_date(forecast_start, series_id))
        return res


class UniformMultiModalLoader(object):

    pass
