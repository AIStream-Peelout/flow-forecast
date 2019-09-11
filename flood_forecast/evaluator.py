import numpy as np 
import pandas as pd
from typing import Callable, Tuple
import sklearn.metrics

def stream_baseline(river_flow_df:pd.DataFrame, forecast_column:str, hours_forecast=336)->(pd.DataFrame, float):
    """
    Function to compute the baseline MSE 
    by using the mean value from the train data. 
    """
    total_length = len(river_flow_df.index)
    train_river_data = river_flow_df[:total_length-hours_forecast]
    test_river_data = river_flow_df[total_length-hours_forecast:]
    mean_value = train_river_data[[forecast_column]].median()[0]
    test_river_data['predicted_baseline'] = mean_value
    mse_baseline = sklearn.metrics.mean_squared_error(test_river_data[forecast_column], test_river_data["predicted_baseline"])
    return test_river_data, round(mse_baseline, ndigits=3)

def plot_r2(river_flow_preds:pd.DataFrame)->float:
    """
    We assume at this point river_flow_preds already has 
    a predicted_baseline and a predicted_model column
    """
    pass

def get_model_score(river_flow_df, model_evaluate_function:Callable, forecast_column:str, hours_forecast=336):
    """

    model_evaluate_function should call any necessary preprocessing
    """
    test_river_data, baseline_mse = stream_baseline(river_flow_df, forecast_column)
