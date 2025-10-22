import pandas as pd
from typing import Dict
import numpy as np


def create_feature(key: str, value: str, df: pd.DataFrame, dt_column: str):
    """Extracts a specific temporal component and applies encoding (numerical or cyclical).

    This is a helper function that extracts a value (like hour or month) from the 
    datetime column and optionally transforms it into sine/cosine components 
    if 'cyclical' encoding is requested.

    :param key: The temporal component to extract (e.g., 'hour', 'month', 'day_of_week').
    :type key: str
    :param value: The type of feature encoding to apply: 'cyclical' or 'numerical'.
    :type value: str
    :param df: The Pandas DataFrame to modify.
    :type df: pd.DataFrame
    :param dt_column: The name of the existing datetime column from which to extract features.
    :type dt_column: str
    :return: The DataFrame with the newly created feature column(s).
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
    """Orchestrates the creation and encoding of multiple temporal features.

    Reads the datetime parameters from the configuration dictionary and iteratively 
    calls 'create_feature' for each requested component, collecting the resulting 
    column names.

    :param preprocess_params: Configuration dictionary containing a 'datetime_params' 
                              key which maps temporal components to their desired encoding 
                              (e.g., {"datetime_params": {"month": "cyclical", "day": "numerical"}}).
    :type preprocess_params: Dict
    :param dt_column: The name of the datetime column in the DataFrame.
    :type dt_column: str
    :param df: The DataFrame to which the new temporal features will be added.
    :type df: pd.DataFrame
    :return: A tuple containing the updated DataFrame and a list of the new column names 
             that were successfully added.
    :rtype: Tuple[pd.DataFrame, List[str]]

    .. code-block:: python
        feats_to_add = {"datetime_params": {"month":"cyclical", "day":"numerical"}}
        df, column_names = feature_fix(feats_to_add, "datetime", df)
        # column_names will be ["cos_month", "sin_month", "day"]
    """
    print("Running the code to add temporal features")
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
    """Creates cyclical sine and cosine features for a given numeric column.

    This technique is essential for representing features like 'hour' or 'month' 
    in a way that a machine learning model understands their circular nature 
    (e.g., 23:00 is temporally close to 00:00).

    :param df: The Pandas DataFrame containing the feature to encode.
    :type df: pd.DataFrame
    :param feature_column: The name of the column containing the temporal feature 
                           (e.g., 'day_of_week', 'hour', 'month').
    :type feature_column: str
    :return: The DataFrame with three new columns: a temporary 'norm' column, and 
             the encoded 'cos_<feature_column>' and 'sin_<feature_column>' columns.
    :rtype: pd.DataFrame
    """
    df["norm"] = 2 * np.pi * df[feature_column] / df[feature_column].max()
    df['cos_' + feature_column] = np.cos(df['norm'])
    df['sin_' + feature_column] = np.sin(df['norm'])
    return df
