from datetime import datetime
import pandas as pd
from typing import Dict


def make_temporal_features(features_list: Dict, dt_column: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create features
    """
    df[dt_column] = df[dt_column].to_datetime()
    for key, value in features_list.items():
        df[key] = df[dt_column].map(value)
    return df


def create_feature(key, value, df, dt_column):
    if key == "day_of_week":
        df[key] = df[dt_column].map(lambda x: x.weekday())
    elif key == "hour":
        df[key] = df[dt_column].map(lambda x: x.hour)
    elif key == "month":
        df[key] = df[dt_column].map(lambda x: x.month)
    elif key == "year":
        df[key] = df[dt_column].map(lambda x: x.year)
    if value == "cyclical":
        df = cyclical(df, key)
    return df


def preprocess_data(preprocess_params, dt_column, df):
    column_names = []
    if "datetime_params" in preprocess_params:
        for key, value in preprocess_params:
            create_feature(key, value, df, dt_column)
    return keys, df

