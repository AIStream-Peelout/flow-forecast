import datetime 
import pandas as pd 
from typing import List, Dict

def make_temporal_features(features_list:Dict, dt_column:str, df:pd.DataFrame)->pd.DataFrame:
    """
    Function to create features
    """
    df[dt_column] = df[dt_column].to_datetime()
    for key, value in features_list.items():
        df[key] = df[dt_column].map(value)
    return df