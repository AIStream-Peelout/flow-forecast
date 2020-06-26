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


def get_day(x: datetime) -> int:
    return x.day


def get_month(x: datetime):
    return x.month


def get_hour(x: datetime):
    return x.hour


def get_weekday(x: datetime):
    return x.weekday()
