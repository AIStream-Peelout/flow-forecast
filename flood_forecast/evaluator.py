import numpy as np 
import pandas as pd
from typing import Callable, Tuple
import sklearn.metrics
from torch

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
    print(mse_baseline)
    return test_river_data, round(mse_baseline, ndigits=3)

def plot_r2(river_flow_preds:pd.DataFrame)->float:
    """
    We assume at this point river_flow_preds already has 
    a predicted_baseline and a predicted_model column
    """
    pass

def get_model_r2_score(river_flow_df, model_evaluate_function:Callable, forecast_column:str, hours_forecast=336):
    """

    model_evaluate_function should call any necessary preprocessing
    """
    test_river_data, baseline_mse = stream_baseline(river_flow_df, forecast_column)

def get_r2_value(model_mse, baseline_mse):
    return 1-model_mse/baseline_mse

def get_value():
    df = pd.read_csv("final_keag.csv")
    res = stream_baseline(df, "cfs", 336)
    print(get_r2_value(0.120, res[1]))

def infer_on_torch_model(model, metric:str, test_df:pd.DataFrame = None, hours_forecast:int = 336): 
    """
    Function to handle both test evaluation and inference on a test dataframe 
    """
    forecast_length = model.params["forecast_length"]
    # If the test dataframe is none use default one supplied in params
    if test_df is None:
        data_loader = DataLoader(model.test, batch_size=1, shuffle=False, sampler=None,
            batch_sampler=None, num_workers=0, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None)

    for i in range(0, hours_forecast/forecast_length):
        pass
        
    
    

