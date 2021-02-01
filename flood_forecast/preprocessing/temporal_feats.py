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


def create_feature(key: str, value: str, df: pd.DataFrame, dt_column: str):
    """Function to create temporal features

    :param key: The datetime feature you would like to create
    :type key: str
    :param value: The type of feature you would like to create (cyclical or numerical)
    :type value: str
    :param df: The Pandas dataframe with the datetime
    :type df: pd.DataFrame
    :param dt_column: The name of the datetime column
    :type dt_column: str
    :return: [description]
    :rtype: [type]
    """
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
    """Adds temporal features

    :param preprocess_params: [description]
    :type preprocess_params: [type]
    :param dt_column: [description]
    :type dt_column: [type]
    :param df: [description]
    :type df: [type]
    :return: [description]
    :rtype: [type]
    """
    print("running feature fix code s")
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


def cyclical(df: pd.DataFrame, feature_column: str) -> pd.DataFrame:
    """ A function to create cyclical encodings for Pandas data-frames.

    :param df: A Pandas Dataframe where you want the dt encoded
    :type df: pd.DataFrame
    :param feature_column: The name of the feature column. Should be
    either (day_of_week, hour, month, year)
    :type feature_column: str
    :return: The dataframew with three new columns: norm_feature, cos_feature
    sin_feature
    :rtype: pd.DataFrame
    """
    df["norm"] = 2 * np.pi * df[feature_column] / df[feature_column].max()
    df['cos_' + feature_column] = np.cos(df['norm'])
    df['sin_' + feature_column] = np.sin(df['norm'])
    return df
