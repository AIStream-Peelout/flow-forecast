import numpy as np 
import pandas as pd
import torch 
from datetime import datetime
from typing import Callable, Tuple, Dict, List, Type
from flood_forecast.time_model import TimeSeriesModel
import sklearn.metrics
from flood_forecast.model_dict_function import decoding_functions
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader
from flood_forecast.utils import flatten_list_function

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

def get_model_r2_score(river_flow_df:pd.DataFrame, model_evaluate_function:Callable, forecast_column:str, hours_forecast=336):
    """

    model_evaluate_function should call any necessary preprocessing
    """
    test_river_data, baseline_mse = stream_baseline(river_flow_df, forecast_column)

def get_r2_value(model_mse, baseline_mse):
    return 1-model_mse/baseline_mse

def get_value(the_path:str)->None:
    df = pd.read_csv(the_path)
    res = stream_baseline(df, "cfs", 336)
    print(get_r2_value(0.120, res[1]))

def metric_dict(metric: str) -> Callable:
    dic = {"MSE": torch.nn.MSELoss(), "L1": torch.nn.L1Loss()}
    return dic[metric]


def evaluate_model(model:Type[TimeSeriesModel], model_type:str, target_col: List[str], evaluation_metrics:List, inference_params:Dict, eval_log:Dict)->Tuple:
    """
    A function to evaluate a model.
    Requires a model of type TimeSeriesModel
    """
    if model_type == "PyTorch":
        df, end_tensor, forecast_history, junk, test_data = infer_on_torch_model(model, **inference_params)
        # Unscale test data if scaler was applied
        print("test_data scale")
        if test_data.scale:
            print("Un-transforming data")
            end_tensor = test_data.inverse_scale(end_tensor.detach().reshape(-1,1))
            end_tensor_list = flatten_list_function(end_tensor.numpy().tolist())
            history_length = model.params["dataset_params"]["forecast_history"]
            df['preds'][history_length:] = end_tensor_list
            end_tensor = end_tensor.squeeze(1)
        print("Current historical dataframe")
        print(df)
    for evaluation_metric in evaluation_metrics:
        for target in target_col:
            evaluation_metric_function = metric_dict(evaluation_metric)
            s = evaluation_metric_function(torch.from_numpy(df[target][forecast_history:].to_numpy()), end_tensor)
            eval_log[target + "_" + evaluation_metric] = s
    return eval_log, df, junk


def infer_on_torch_model(model, test_csv_path:str = None, datetime_start=datetime(2018, 9, 22, 0), hours_to_forecast: int = 336, decoder_params=None, dataset_params:Dict={}): 
    """
    Function to handle both test evaluation and inference on a test dataframe. 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type(datetime_start) == str:
        datetime_start = datetime.strptime(datetime_start, '%Y-%m-%d')
    history_length = model.params["dataset_params"]["forecast_history"]
    forecast_length = model.params["dataset_params"]["forecast_length"]
    # If the test dataframe is none use default one supplied in params
    if test_csv_path is None:
        test_data = model.test_data
    else:
        test_data = CSVTestLoader(test_csv_path, hours_to_forecast, **dataset_params, interpolate=dataset_params["interpolate_param"])
    model.model.eval()
    history, df, forecast_start_idx = test_data.get_from_start_date(datetime_start)
    all_tensor = []
    history_dim = history.unsqueeze(0).to(model.device)
    print(history_dim.shape)
    print("Add debugging crap below")
    real_target_tensor = torch.from_numpy(test_data.df[forecast_start_idx:].to_numpy()).to(device).unsqueeze(0).to(model.device)
    full_history = [history_dim]
    if test_data.use_real_precip:
        precip_cols = test_data.convert_real_batches('precip', df[forecast_length:])
    if test_data.use_real_temp:
        temp_cols = test_data.convert_real_batches('temp', df[forecast_length:])
    if decoder_params is None:
        for i in range(0, int(np.ceil(hours_to_forecast/forecast_length).item())):
            output = model.model(full_history[i].to(model.device))
            all_tensor.append(output.view(-1))
            if i == int(np.ceil(hours_to_forecast/forecast_length).item())-1:
                break
            rel_cols = model.params["dataset_params"]["relevant_cols"]
            if test_data.use_real_precip and test_data.use_real_temp:
                # Order here should match order of original tensor... But what is the best way todo that...?
                # Hmm right now this will create a bug if for some reason the order [precip, temp, output]
                intial_numpy = torch.stack([output.view(-1).float().to(model.device), precip_cols[i].float().to(model.device), temp_cols[i].float().to(model.device)]).to('cpu').detach().numpy()
                temp_df = pd.DataFrame(intial_numpy.T, columns=rel_cols)
                revised_np = temp_df[rel_cols].to_numpy()
                full_history.append(torch.from_numpy(revised_np).to(model.device).unsqueeze(0))
        remainder = forecast_length - hours_to_forecast % forecast_length
        if remainder != forecast_length:
            # Subtract remainder from array
            end_tensor = torch.cat(all_tensor, axis=0).to('cpu').detach()[:-remainder]
        else:
            end_tensor = torch.cat(all_tensor, axis=0).to('cpu').detach()

        print(end_tensor.shape)
    else:
        # model, src, max_seq_len, real_target, output_len=1, unsqueeze_dim=1
        # hours_to_forecast 336 
        #greedy_decode(model, src, sequence_size, targ, src, device=device)[:, :, 0]
        # greedy_decode(model, src:torch.Tensor, max_len:int, real_target:torch.Tensor, start_symbol:torch.Tensor, unsqueeze_dim=1, device='cpu')
        end_tensor = decoding_functions[decoder_params["decoder_function"]](model.model, history_dim, hours_to_forecast, real_target_tensor, decoder_params["unsqueeze_dim"], device=model.device)
        end_tensor = end_tensor[:, :, 0].view(-1).to('cpu').detach()
    df['preds'] = 0
    df['preds'][history_length:] = end_tensor.numpy().tolist()
    print(end_tensor.shape)
    return df, end_tensor, history_length, forecast_start_idx, test_data