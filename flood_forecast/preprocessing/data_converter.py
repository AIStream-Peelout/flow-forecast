"""A set of function aimed at making it easy to convert other time series datasets to our format for transfer learning
purposes."""

import pandas as pd

def make_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Renames all columns in a DataFrame using a standard "solar_i" format,
    where 'i' is the column's zero-based index.

    :param df: The input pandas DataFrame with existing column names.
    :type df: pd.DataFrame
    :return: The DataFrame with standardized column names.
    :rtype: pd.DataFrame
    """
    num_cols = len(list(df))
    # generate range of ints for suffixes
    # with length exactly half that of num_cols;
    # if num_cols is even, truncate concatenated list later
    # to get to original list length
    column_arr = []
    for i in range(0, num_cols):
        column_arr.append("solar_" + str(i))

    # ensure the length of the new columns list is equal to the length of df's columns
    df.columns = column_arr
    return df