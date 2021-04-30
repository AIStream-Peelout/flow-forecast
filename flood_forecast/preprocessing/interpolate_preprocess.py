import pandas as pd
from typing import List


def fix_timezones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic function to fix initil data bug
    related to NaN values in non-eastern-time zones due
    to UTC conversion.
    """
    the_count = df[0:2]['cfs'].isna().sum()
    return df[the_count:]


def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to fill missing values with nearest value.
    Should be run only after splitting on the NaN chunks.
    """
    df = fix_timezones(df)
    df['cfs1'] = df['cfs'].interpolate(method='nearest').ffill().bfill()
    df['precip'] = df['p01m'].interpolate(method='nearest').ffill().bfill()
    df['temp'] = df['tmpf'].interpolate(method='nearest').ffill().bfill()
    return df


def forward_back_generic(df: pd.DataFrame, relevant_columns: List) -> pd.DataFrame:
    """
    Function to fill missing values with nearest value (forward first)
    """
    for col in relevant_columns:
        df[col] = df[col].interpolate(method='nearest').ffill().bfill()
    return df


def back_forward_generic(df: pd.DataFrame, relevant_columns: List[str]) -> pd.DataFrame:
    """
    Function to fill missing values with nearest values (backward first)
    """
    for col in relevant_columns:
        df[col] = df[col].interpolate(method='nearest').bfill().ffill()
    return df
