import pandas as pd
from typing import List


def fix_timezones(df: pd.DataFrame) -> pd.DataFrame:
    """Basic function to fix initial data bug related to NaN values in non-eastern-time zones due to UTC conversion.

    It removes leading rows that have NaN values in the 'cfs' column, which can occur after UTC conversion for certain time zones.

    :param df: The input pandas DataFrame, expected to have a 'cfs' column.
    :type df: pd.DataFrame
    :return: The DataFrame with leading NaN 'cfs' rows removed.
    :rtype: pd.DataFrame
    """
    the_count = df[0:2]['cfs'].isna().sum()
    return df[the_count:]


def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Function to fill missing values in 'cfs', 'p01m', and 'tmpf' columns with the nearest available value.

    This should be run after splitting the data on large NaN chunks. The process is: fix timezones, then interpolate using 'nearest' method, followed by forward fill (`ffill`) and backward fill (`bfill`).

    :param df: The input pandas DataFrame, expected to have 'cfs', 'p01m', and 'tmpf' columns.
    :type df: pd.DataFrame
    :return: The DataFrame with new interpolated columns ('cfs1', 'precip', 'temp').
    :rtype: pd.DataFrame
    """
    df = fix_timezones(df)
    df['cfs1'] = df['cfs'].interpolate(method='nearest').ffill().bfill()
    df['precip'] = df['p01m'].interpolate(method='nearest').ffill().bfill()
    df['temp'] = df['tmpf'].interpolate(method='nearest').ffill().bfill()
    return df


def forward_back_generic(df: pd.DataFrame, relevant_columns: List[str]) -> pd.DataFrame:
    """Function to fill missing values in specified columns using nearest interpolation, followed by forward fill then backward fill.

    :param df: The input pandas DataFrame.
    :type df: pd.DataFrame
    :param relevant_columns: A list of column names in the DataFrame to apply the imputation to.
    :type relevant_columns: List[str]
    :return: The DataFrame with missing values imputed in the relevant columns.
    :rtype: pd.DataFrame
    """
    for col in relevant_columns:
        df[col] = df[col].interpolate(method='nearest').ffill().bfill()
    return df


def back_forward_generic(df: pd.DataFrame, relevant_columns: List[str]) -> pd.DataFrame:
    """Function to fill missing values in specified columns using nearest interpolation, followed by backward fill then forward fill.

    :param df: The input pandas DataFrame.
    :type df: pd.DataFrame
    :param relevant_columns: A list of column names in the DataFrame to apply the imputation to.
    :type relevant_columns: List[str]
    :return: The DataFrame with missing values imputed in the relevant columns.
    :rtype: pd.DataFrame
    """
    for col in relevant_columns:
        df[col] = df[col].interpolate(method='nearest').bfill().ffill()
    return df