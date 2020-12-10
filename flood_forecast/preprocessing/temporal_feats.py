import pandas as pd
from typing import Dict
import numpy as np


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


def feature_fix(preprocess_params, dt_column, df):
    print("running feature fix code")
    column_names = []
    if "datetime_params" in preprocess_params:
        for key, value in preprocess_params["datetime_params"].items():
            df = create_feature(key, value, df, dt_column)
            if value == "cyclical":
                column_names.append("cos_" + key)
                column_names.append("sin_" + key)
            else:
                column_names.append(key)
    return df, column_names


def cyclical(df, feature_column):
    df["norm"] = 2 * np.pi * df[feature_column] / df[feature_column].max()
    df['cos_' + feature_column] = np.cos(df['norm'])
    df['sin_' + feature_column] = np.sin(df['norm'])
    return df
