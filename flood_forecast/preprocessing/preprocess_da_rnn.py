import numpy as np
import pandas as pd
from typing import List, NamedTuple, Tuple, Optional


class TrainData(NamedTuple):
    """
    A named tuple to hold feature and target NumPy arrays for training.

    :ivar features: NumPy array of features (X).
    :vartype features: np.ndarray
    :ivar targets: NumPy array of targets (Y).
    :vartype targets: np.ndarray
    """
    features: np.ndarray
    targets: np.ndarray


def format_data(dat: pd.DataFrame, targ_column: List[str]) -> TrainData:
    """Converts a pandas DataFrame into a structured format of features and targets (NumPy arrays).

    The columns specified in `targ_column` are separated as targets, and the remaining columns are used as features.

    :param dat: The input DataFrame containing both features and targets.
    :type dat: pd.DataFrame
    :param targ_column: A list of column names that should be treated as targets.
    :type targ_column: List[str]
    :return: A TrainData NamedTuple containing the separated features and targets as NumPy arrays.
    :rtype: TrainData
    """
    # Test numpy conversion
    proc_dat = dat.to_numpy()
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in targ_column:
        # Create a boolean mask where True means 'feature' and False means 'target'
        # Set the target column indices to False
        mask[dat_cols.index(col_name)] = False

    # Apply the mask: select columns where mask is True for features (X)
    feats = proc_dat[:, mask].astype(float)
    # Apply the inverse mask: select columns where mask is False for targets (Y)
    targs = proc_dat[:, ~mask].astype(float)
    return TrainData(feats, targs)


def make_data(
    csv_path: str,
    target_col: List[str],
    test_length: int,
    relevant_cols: List[str] = [
        "cfs",
        "temp",
        "precip"]) -> TrainData:
    """Reads a CSV file, selects relevant columns, and formats the data into features and targets.

    This function does not perform the train/test split; it returns the full preprocessed dataset.

    :param csv_path: The file path to the input CSV data.
    :type csv_path: str
    :param target_col: A list of column names that should be used as prediction targets.
    :type target_col: List[str]
    :param test_length: The length of the test set (used for reference, but not for splitting the data here).
    :type test_length: int
    :param relevant_cols: A list of feature columns to include if only one target column is specified. Defaults to ["cfs", "temp", "precip"].
    :type relevant_cols: List[str]
    :return: The full preprocessed data as a TrainData NamedTuple.
    :rtype: TrainData
    """
    final_df = pd.read_csv(csv_path)
    print(final_df.shape[0])

    if len(target_col) > 1:
        # If multiple targets, specifically select them and the default features
        height_df = final_df[[target_col[0], target_col[1], 'precip', 'temp']]
        height_df.columns = [target_col[0], target_col[1], 'precip', 'temp']
    else:
        # If one target, combine it with the specified relevant_cols
        height_df = final_df[[target_col[0]] + relevant_cols]

    preprocessed_data2 = format_data(height_df, target_col)
    return preprocessed_data2