import pandas as pd
from typing import Dict
import numpy as np


def create_feature(key: str, value: str, df: pd.DataFrame, dt_column: str):
    """Function to create temporal feature. Uses dict to make val.

    :param key: The datetime feature you would like to create
    :type key: str
    :param value: The type of feature you would like to create (cyclical or numerical)
    :type value: str
    :param df: The Pandas dataframe with the datetime
    :type df: pd.DataFrame
    :param dt_column: The name of the datetime column
    :type dt_column: str
    :return: The dataframe with the newly added column
    :rtype: pd.DataFrame
    """
    if key == "day_of_week":
        df[key] = df[dt_column].map(lambda x: x.weekday())
    elif key == "minute":
        df[key] = df[dt_column].map(lambda x: x.minute)
    elif key == "hour":
        df[key] = df[dt_column].map(lambda x: x.hour)
    elif key == "day":
        df[key] = df[dt_column].map(lambda x: x.day)
    elif key == "month":
        df[key] = df[dt_column].map(lambda x: x.month)
    elif key == "year":
        df[key] = df[dt_column].map(lambda x: x.year)
    if value == "cyclical":
        df = cyclical(df, key)
    return df


def feature_fix(preprocess_params: Dict, dt_column: str, df: pd.DataFrame):
    """Adds temporal features

    :param preprocess_params: Dictionary of temporal parameters e.g. {"day":"numerical"}
    :type preprocess_params: Dict
    :param dt_column: The column name of the data
    :param df: The dataframe to add the temporal features to
    :type df: pd.DataFrame
    :return: Returns the new data-frame and a list of the new column names
    :rtype: Tuple(pd.Dataframe, List[str])

    .. code-block:: python
        feats_to_add = {"month":"cyclical", "day":"numerical"}
        df, column_names feature_fix(feats_to_add, "datetime")
        print(column_names) # ["cos_month", "sin_month", "day"]
    """
    print("Running code to add temporal features")
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
    :param feature_column: The name of the feature column. Should be either (day_of_week, hour, month, year)
    :type feature_column: str
    :return: The dataframe with three new columns: norm_feature, cos_feature, sin_feature
    :rtype: pd.DataFrame
    """
    df["norm"] = 2 * np.pi * df[feature_column] / df[feature_column].max()
    df['cos_' + feature_column] = np.cos(df['norm'])
    df['sin_' + feature_column] = np.sin(df['norm'])
    return df
