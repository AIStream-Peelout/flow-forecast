import numpy as np 
import pandas as pd
import torch 
from datetime import datetime
from typing import Callable, Tuple, Dict, List
import sklearn.metrics
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader

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

def metric_dict(metric:str):
    dic = {"MSE": torch.nn.MSELoss(), "L1": torch.nn.L1Loss()}
    return dic[metric]


def evaluate_model(model, model_type:str, target_col:str, evaluation_metrics:List, inference_params:Dict, eval_log:Dict):
    """
    A function to evaluate a model
    Requires a model of type
    """
    if model_type == "PyTorch":
        df, end_tensor, forecast_history, junk = infer_on_torch_model(model, **inference_params)
    for evaluation_metric in evaluation_metrics:
        evaluation_metric_function = metric_dict(evaluation_metric)
        print('frame data')
        print(df[target_col][forecast_history:])
        s = evaluation_metric_function(torch.from_numpy(df[target_col][forecast_history:].to_numpy()), torch.from_numpy(df["preds"][forecast_history:].to_numpy()))
        eval_log[evaluation_metric] = s
    return eval_log

def infer_on_torch_model(model, test_csv_path:str = None, datetime_start=datetime(2018,9,22,0), hours_to_forecast:int = 336, dataset_params:Dict={}): 
    """
    Function to handle both test evaluation and inference on a test dataframe 
    """
    history_length = model.params["dataset_params"]["forecast_history"]
    forecast_length = model.params["dataset_params"]["forecast_length"]
    # If the test dataframe is none use default one supplied in params
    if test_csv_path is None:
        test_data = model.test_data
    else:
        test_data = CSVTestLoader(test_csv_path, hours_to_forecast, **dataset_params)
    model.model.eval()
    history, df, forecast_start_idx = test_data.get_from_start_date(datetime_start)
    all_tensor = []
    full_history = [history.unsqueeze(0)]
    if test_data.use_real_precip:
        precip_cols = test_data.convert_real_batches('precip', df[forecast_length:])
    if test_data.use_real_temp:
        temp_cols = test_data.convert_real_batches('temp', df[forecast_length:])
    for i in range(0, int(np.ceil(hours_to_forecast/forecast_length).item())):
        output = model.model(full_history[i])
        all_tensor.append(output.view(-1))
        if i==int(np.ceil(hours_to_forecast/forecast_length).item())-1:
            break
        if test_data.use_real_precip and test_data.use_real_temp:
            # Order here should match order of original tensor... But what is the best way todo that...? 
            # Hmm right now this will create a bug if for some reason the order [precip, temp, output]
            intial_numpy = torch.stack([output.view(-1).float(), precip_cols[i].float(), temp_cols[i].float()]).to('cpu').detach().numpy()
            temp_df = pd.DataFrame(intial_numpy.T, columns=['cfs', 'precip', 'temp'])
            revised_np = temp_df[model.params["dataset_params"]["relevant_cols"]].to_numpy()
            full_history.append(torch.from_numpy(revised_np).to(model.device).unsqueeze(0))
    remainder = forecast_length - hours_to_forecast % forecast_length
    # Subtract remainder from array
    end_tensor = torch.cat(all_tensor, axis = 0).to('cpu').detach().numpy()[:-remainder]
    df['preds'] = 0
    df['preds'][history_length:] = end_tensor
    return df, end_tensor, history_length, forecast_start_idx
    
        
    
    

