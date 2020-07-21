"""
A set of function aimed at making it easy
to convert other time series datasets to our
format for transfer learning purposes
"""


def make_column_names(df):
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
