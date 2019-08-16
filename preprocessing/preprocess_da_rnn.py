from typing import Tuple
import typing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray
      
    
def format_data(dat, targ_column) -> Tuple[TrainData, StandardScaler]:
    scale = StandardScaler().fit(dat)
  
    #proc_dat = scale.transform(dat)
    #print(proc_dat)
    proc_dat = dat.as_matrix()
    print(proc_dat)
    
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in targ_column:
        mask[dat_cols.index(col_name)] = False
    feats = proc_dat[:, mask].astype(float)
    targs = proc_dat[:, ~mask].astype(float)
    return TrainData(feats, targs), scale

def make_data(csv_path:str, target_col:str)->TrainData:
    final_df = pd.read_csv("sebring.csv")
    # Doing this a temporary fix please comment in or out.
    final_df = final_df[:6087]
    height_df = final_df[['height', 'precip', 'temp']]
    height_df.columns = ['height', 'precip', 'temp']
    preprocessed_data2, s = format_data(height_df, ['height'])
    return preprocessed_data2


