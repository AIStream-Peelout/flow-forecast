import numpy as np
import pandas as pd
from typing import List
from flood_forecast.da_rnn.custom_types import TrainData


def format_data(dat, targ_column: List[str]) -> TrainData:
    # Test numpy conversion
    proc_dat = dat.to_numpy()
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in targ_column:
        mask[dat_cols.index(col_name)] = False
    feats = proc_dat[:, mask].astype(float)
    targs = proc_dat[:, ~mask].astype(float)
    return TrainData(feats, targs)


def make_data(
    csv_path: str,
    target_col: List[str],
    test_length: int,
    relevant_cols=[
        "cfs",
        "temp",
        "precip"]) -> TrainData:
    """
    Returns full preprocessed data.
    Does not split train/test that must be done later.
    """
    final_df = pd.read_csv(csv_path)
    print(final_df.shape[0])
    if len(target_col) > 1:
        # Restrict target columns to height and cfs. Alternatively could replace this with loop
        height_df = final_df[[target_col[0], target_col[1], 'precip', 'temp']]
        height_df.columns = [target_col[0], target_col[1], 'precip', 'temp']
    else:
        height_df = final_df[[target_col[0]] + relevant_cols]
    preprocessed_data2 = format_data(height_df, target_col)
    return preprocessed_data2
