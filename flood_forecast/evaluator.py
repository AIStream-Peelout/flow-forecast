from datetime import datetime
from typing import Callable, Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
import sklearn.metrics
import torch

from flood_forecast.explain_model_output import (
    deep_explain_model_heatmap,
    deep_explain_model_summary_plot,
)
from flood_forecast.model_dict_function import decoding_functions
from flood_forecast.custom.custom_opt import MASELoss, GaussianLoss
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader, TemporalTestLoader
from flood_forecast.time_model import TimeSeriesModel
from flood_forecast.utils import flatten_list_function
from flood_forecast.temporal_decoding import decoding_function


def stream_baseline(
    river_flow_df: pd.DataFrame, forecast_column: str, hours_forecast=336
) -> Tuple[pd.DataFrame, float]:
    """
    Function to compute the baseline MSE
    by using the mean value from the train data.
    """
    total_length = len(river_flow_df.index)
    train_river_data = river_flow_df[: total_length - hours_forecast]
    test_river_data = river_flow_df[total_length - hours_forecast:]
    mean_value = train_river_data[[forecast_column]].median()[0]
    test_river_data["predicted_baseline"] = mean_value
    mse_baseline = sklearn.metrics.mean_squared_error(
        test_river_data[forecast_column], test_river_data["predicted_baseline"]
    )
    return test_river_data, round(mse_baseline, ndigits=3)


def plot_r2(river_flow_preds: pd.DataFrame) -> float:
    """
    We assume at this point river_flow_preds already has
    a predicted_baseline and a predicted_model column
    """
    pass


def get_model_r2_score(
    river_flow_df: pd.DataFrame,
    model_evaluate_function: Callable,
    forecast_column: str,
    hours_forecast=336,
):
    """

    model_evaluate_function should call any necessary preprocessing
    """
    test_river_data, baseline_mse = stream_baseline(river_flow_df, forecast_column)


def get_r2_value(model_mse, baseline_mse):
    return 1 - model_mse / baseline_mse


def get_value(the_path: str) -> None:
    df = pd.read_csv(the_path)
    res = stream_baseline(df, "cfs", 336)
    print(get_r2_value(0.120, res[1]))


def evaluate_model(
    model: Type[TimeSeriesModel],
    model_type: str,
    target_col: List[str],
    evaluation_metrics: List,
    inference_params: Dict,
    eval_log: Dict,
) -> Tuple[Dict, pd.DataFrame, int, pd.DataFrame]:
    """
    A function to evaluate a model. Called automatically at end of training.
    Can be imported for continuing to evaluate a model in other places as well.


    .. highlight:: python
    .. code-block:: python

        from flood_forecast.evaluator import evaluate_model
        forecast_model = PyTorchForecast(config_file)
        e_log, df_train_test, f_idx, df_preds = evaluate_model(forecast_model, "PyTorch", ["cfs"], ["MSE", "MAPE"], {})
        print(e_log) # {"MSE":0.2, "MAPE":0.1}
        print(df_train_test) # will print a pandas dataframe
        ...
    '''
    """
    if model_type == "PyTorch":
        (
            df_train_and_test,
            end_tensor,
            forecast_history,
            forecast_start_idx,
            test_data,
            df_predictions,
            # df_prediction_samples_std_dev,
        ) = infer_on_torch_model(model, **inference_params)
        # To-do turn this into a general function
        g_loss = False
        probablistic = True if "probabilistic" in inference_params else False
        if isinstance(end_tensor, tuple) and not probablistic:
            end_tensor_0 = end_tensor[1]
            end_tensor = end_tensor[0]
            g_loss = True
        if test_data.scale:
            print("Un-transforming data")
            if probablistic:
                print('probabilistic running on infer_on_torch_model')
                end_tensor_mean = test_data.inverse_scale(end_tensor[0].detach().reshape(-1, 1))
                end_tensor_list = flatten_list_function(end_tensor_mean.numpy().tolist())
                end_tensor_mean = end_tensor_mean.squeeze(1)
            else:
                if "n_targets" in model.params:
                    if model.params["model_name"] == "Informer":
                        end_tensor = end_tensor[:, :, 0:model.params["n_targets"]]
                    end_tensor = test_data.inverse_scale(end_tensor.detach())
                else:
                    end_tensor = test_data.inverse_scale(end_tensor.detach().reshape(-1, 1))
                end_tensor_list = flatten_list_function(end_tensor.numpy().tolist())
                end_tensor = end_tensor.squeeze(1)  # Removing extra dim from reshape?
            history_length = model.params["dataset_params"]["forecast_history"]
            if "n_targets" in model.params:
                df_train_and_test.loc[df_train_and_test.index[history_length:],
                                      "preds"] = end_tensor[:, 0].numpy().tolist()
                for i, target in enumerate(target_col):
                    df_train_and_test["pred_" + target] = 0
                    df_train_and_test.loc[df_train_and_test.index[history_length:],
                                          "pred_" + target] = end_tensor[:, i].numpy().tolist()
            else:
                df_train_and_test.loc[df_train_and_test.index[history_length:], "preds"] = end_tensor_list
                df_train_and_test["pred_" + target_col[0]] = 0
                df_train_and_test.loc[df_train_and_test.index[history_length:],
                                      "pred_" + target_col[0]] = end_tensor_list
        print("Current historical dataframe ")
        print(df_train_and_test)
    for evaluation_metric in model.crit:
        idx = 0
        for target in target_col:
            labels = torch.from_numpy(df_train_and_test[target][forecast_history:].to_numpy())
            evaluation_metric_function = evaluation_metric
            if "probabilistic" in inference_params:
                s = evaluation_metric_function(
                    torch.distributions.Normal(end_tensor[0], end_tensor[1][0]),
                    labels,
                )
            elif isinstance(evaluation_metric_function, MASELoss):
                s = evaluation_metric_function(
                    labels,
                    end_tensor,
                    torch.from_numpy(
                        df_train_and_test[target][:forecast_history].to_numpy()
                    )
                )
            elif g_loss:
                g = GaussianLoss(end_tensor.unsqueeze(1), end_tensor_0.unsqueeze(1))
                s = g(labels.unsqueeze(1))

            else:
                if "n_targets" in model.params:
                    s = evaluation_metric_function(
                        labels,
                        end_tensor[:, idx],
                    )
                else:
                    s = evaluation_metric_function(
                        labels,
                        end_tensor,
                    )
            idx += 1
            eval_log[target + "_" + evaluation_metric.__class__.__name__] = s

    # Explain model behaviour using shap
    if "probabilistic" in inference_params:
        print("Probabilistic explainability currently not supported.")
    elif "n_targets" in model.params:
        print("Multitask forecasting support coming soon")
    elif g_loss:
        print("SHAP not yet supported for these models with multiple outputs")
    else:
        deep_explain_model_summary_plot(
            model, test_data, inference_params["datetime_start"]
        )
        deep_explain_model_heatmap(model, test_data, inference_params["datetime_start"])

    return eval_log, df_train_and_test, forecast_start_idx, df_predictions


def infer_on_torch_model(
    model,
    test_csv_path: str = None,
    datetime_start: datetime = datetime(2018, 9, 22, 0),
    hours_to_forecast: int = 336,
    decoder_params=None,
    dataset_params: Dict = {},
    num_prediction_samples: int = None,
    probabilistic: bool = False,
    criterion_params: Dict = None
) -> Tuple[pd.DataFrame, torch.Tensor, int, int, CSVTestLoader, List[pd.DataFrame]]:
    """
    Function to handle both test evaluation and inference on a test data-frame.
    :param model: The time series model present
    :param test_csv_path: The path to the test data-frame
    :return:
        df: df including training and test data
        end_tensor: the final tensor after the model has finished predictions
        history_length: num rows to use in training
        forecast_start_idx: row index to start forecasting
        test_data: CSVTestLoader instance
        df_prediction_samples: has same index as df, and num cols equal to num_prediction_samples
        or no columns if num_prediction_samples is None
    :rtype: tuple()
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(datetime_start, str):
        datetime_start = datetime.strptime(datetime_start, "%Y-%m-%d")
    multi_params = 1
    if "n_targets" in model.params:
        multi_params = model.params["n_targets"]
    print("This model is currently forecasting for: " + str(multi_params) + " targets")
    history_length = model.params["dataset_params"]["forecast_history"]
    forecast_length = model.params["dataset_params"]["forecast_length"]
    sort_column2 = None
    #
    # If the test dataframe is none use default one supplied in params
    if test_csv_path is None:
        csv_test_loader = model.test_data
    elif model.params["dataset_params"]["class"] == "TemporalLoader":
        input_dict = {
            "df_path": test_csv_path,
            "forecast_total": hours_to_forecast,
            "kwargs": dataset_params
        }
        test_idx = None
        if "label_len" in model.params["model_params"]:
            test_idx = model.params["model_params"]["label_len"] - model.params["dataset_params"]["forecast_length"]
        csv_test_loader = TemporalTestLoader(model.params["dataset_params"]["temporal_feats"], input_dict, test_idx)
    else:
        csv_test_loader = CSVTestLoader(
            test_csv_path,
            hours_to_forecast,
            **dataset_params,
            sort_column_clone=sort_column2,
            interpolate=dataset_params["interpolate_param"]
        )
    model.model.eval()
    targ = False
    if model.params["dataset_params"]["class"] == "TemporalLoader":
        history, targ, df_train_and_test, forecast_start_idx = csv_test_loader.get_from_start_date(datetime_start)
    else:
        (
            history,
            df_train_and_test,
            forecast_start_idx,
        ) = csv_test_loader.get_from_start_date(datetime_start)
    end_tensor = generate_predictions(
        model,
        df_train_and_test,
        csv_test_loader,
        history,
        device,
        forecast_start_idx,
        forecast_length,
        hours_to_forecast,
        decoder_params,
        multi_params=multi_params,
        targs=targ
    )

    df_train_and_test["preds"] = 0
    if decoder_params is not None:
        if "probabilistic" in decoder_params:
            df_train_and_test.loc[df_train_and_test.index[history_length:], "preds"] = end_tensor[0].numpy().tolist()
            df_train_and_test["std_dev"] = 0
            print('end_tensor[1][0].numpy().tolist()', end_tensor[1][0].numpy().tolist())
            try:
                df_train_and_test.loc[df_train_and_test.index[history_length:],
                                      "std_dev"] = end_tensor[1][0].numpy().tolist()
            except Exception as e:
                df_train_and_test.loc[df_train_and_test.index[history_length:],
                                      "std_dev"] = [x[0] for x in end_tensor[1][0].numpy().tolist()]
                print(e)
    else:
        df_train_and_test.loc[df_train_and_test.index[history_length:], "preds"] = end_tensor.numpy().tolist()
    df_prediction_arr = []
    df_prediction_samples = pd.DataFrame(index=df_train_and_test.index)
    # df_prediction_samples_std_dev = pd.DataFrame(index=df_train_and_test.index)
    if num_prediction_samples is not None:
        model.model.train()  # sets mode to train so the dropout layers will be touched
        assert num_prediction_samples > 0
        prediction_samples = generate_prediction_samples(
            model,
            df_train_and_test,
            csv_test_loader,
            history,
            device,
            forecast_start_idx,
            forecast_length,
            hours_to_forecast,
            decoder_params,
            num_prediction_samples,
            multi_params=multi_params,
            targs=targ
        )
        df_prediction_samples = pd.DataFrame(
            index=df_train_and_test.index,
            columns=list(range(num_prediction_samples)),
            dtype="float",
        )
        num_samples = model.params["inference_params"].get("num_prediction_samples")
        df_prediction_arr = handle_ci_multi(prediction_samples, csv_test_loader, multi_params,
                                            df_prediction_samples, decoder_params, history_length, num_samples)
    return (
        df_train_and_test,
        end_tensor,
        history_length,
        forecast_start_idx,
        csv_test_loader,
        df_prediction_arr,
        # df_prediction_samples_std_dev
    )


def handle_ci_multi(prediction_samples: torch.Tensor, csv_test_loader: CSVTestLoader, multi_params: int,
                    df_pred, decoder_param: bool, history_length: int, num_samples: int) -> List[pd.DataFrame]:
    """Handles the CI confidence interval

    :param prediction_samples: The number of predictions to generate
    :type prediction_samples: torch.Tensor
    :param csv_test_loader: The test loader genreated in the previous
    :type csv_test_loader: CSVTestLoader
    :param multi_params: [description]
    :type multi_params: int
    :param df_pred: [description]
    :type df_pred: [type]
    :param decoder_param: [description]
    :type decoder_param: bool
    :param history_length: [description]
    :type history_length: int
    :param num_samples: [description]
    :type num_samples: int
    :raises ValueError: [description]
    :raises ValueError: [description]
    :return: Returns an array with different CI predictions
    :rtype: List[pd.DataFrame]
    """
    df_prediction_arr = []
    if decoder_param is not None:
        if "probabilistic" in decoder_param:
            prediction_samples = prediction_samples[0]
        if multi_params == 1:
            print(type(prediction_samples))
            predict = csv_test_loader.inverse_scale(prediction_samples).numpy()
            prediction_samples = predict
            df_pred.iloc[history_length:] = prediction_samples
            df_prediction_arr.append(df_pred)
        else:
            print(prediction_samples.shape)
            for i in range(0, num_samples):
                tra = prediction_samples[:, :, 0, i]
                prediction_samples[:, :, 0, i] = csv_test_loader.inverse_scale(tra.transpose(1, 0)).transpose(1, 0)
                if i > 0:
                    if np.equal(tra, prediction_samples[:, :, 0, i - 1]).all():
                        print("WARNING model values are the same. Try varying dropout or other mechanism")
            for i in range(0, multi_params):
                if i > 0:
                    if np.equal(prediction_samples[i, :, 0, :], prediction_samples[i - 1, :, 0, :]).all():
                        raise ValueError("Something is wrong data for the targets is equal")
                df_pred.iloc[history_length:] = prediction_samples[i, :, 0, :]
                df_prediction_arr.append(df_pred.copy())
    else:
        df_pred.iloc[history_length:] = prediction_samples
        df_prediction_arr.append(df_pred)
    if len(df_prediction_arr) < 1:
        raise ValueError("Error length of the prediction array must be one or greater")
    return df_prediction_arr


def generate_predictions(
    model: Type[TimeSeriesModel],
    df: pd.DataFrame,
    test_data: CSVTestLoader,
    history: torch.Tensor,
    device: torch.device,
    forecast_start_idx: int,
    forecast_length: int,
    hours_to_forecast: int,
    decoder_params: Dict,
    targs=False,
    multi_params: int = 1
) -> torch.Tensor:
    """A function to generate the actual model prediction

    :param model: A PyTorchForecast
    :type model: Type[TimeSeriesModel]
    :param df: The main dataframe containing data
    :type df: pd.DataFrame
    :param test_data: The test data loader
    :type test_data: CSVTestLoader
    :param history: The forecast historical data
    :type history: torch.Tensor
    :param device: The device usually cpu or cuda
    :type device: torch.device
    :param forecast_start_idx: The index you want the forecast to begin
    :type forecast_start_idx: int
    :param forecast_length: The length of the forecast the model outputs per forward pass
    :type forecast_length: int
    :param hours_to_forecast: The number of time_steps to forecast in future
    :type hours_to_forecast: int
    :param decoder_params: The parameters the decoder function takes.
    :type decoder_params: Dict
    :param multi_params: n_targets, defaults to 1
    :type multi_params: int, optional
    :return: The forecasted values for the time-series in a tensor
    :rtype: torch.Tensor
    """
    if targs:
        history_dim = history
    else:
        history_dim = history.unsqueeze(0).to(model.device)
    print("Add debugging crap below")
    if decoder_params is None:
        end_tensor = generate_predictions_non_decoded(
            model, df, test_data, history_dim, forecast_length, hours_to_forecast,
        )
    else:
        # model, src, max_seq_len, real_target, output_len=1, unsqueeze_dim=1
        # hours_to_forecast 336
        # greedy_decode(model, src, sequence_size, targ, src, device=device)[:, :, 0]
        # greedy_decode(model, src:torch.Tensor, max_len:int,
        # real_target:torch.Tensor, start_symbol:torch.Tensor,
        # unsqueeze_dim=1, device='cpu')
        end_tensor = generate_decoded_predictions(
            model,
            test_data,
            forecast_start_idx,
            device,
            history_dim,
            hours_to_forecast,
            decoder_params,
            multi_targets=multi_params,
            targs=targs
        )
    return end_tensor


def generate_predictions_non_decoded(
    model: Type[TimeSeriesModel],
    df: pd.DataFrame,
    test_data: CSVTestLoader,
    history_dim: torch.Tensor,
    forecast_length: int,
    hours_to_forecast: int,
) -> torch.Tensor:
    """Generates predictions for the models that do not use a decoder

    :param model: A PyTorchForecast
    :type model: Type[TimeSeriesModel]
    :param df: [description]
    :type df: pd.DataFrame
    :param test_data: [description]
    :type test_data: CSVTestLoader
    :param history_dim: [description]
    :type history_dim: torch.Tensor
    :param forecast_length: [description]
    :type forecast_length: int
    :param hours_to_forecast: [description]
    :type hours_to_forecast: int
    :return: [description]
    :rtype: torch.Tensor
    """
    full_history = [history_dim]
    all_tensor = []
    if test_data.use_real_precip:
        precip_cols = test_data.convert_real_batches("precip", df[forecast_length:])
    if test_data.use_real_temp:
        temp_cols = test_data.convert_real_batches("temp", df[forecast_length:])
    for i in range(0, int(np.ceil(hours_to_forecast / forecast_length).item())):
        output = model.model(full_history[i].to(model.device))
        all_tensor.append(output.view(-1))
        if i == int(np.ceil(hours_to_forecast / forecast_length).item()) - 1:
            break
        rel_cols = model.params["dataset_params"]["relevant_cols"]
        if test_data.use_real_precip and test_data.use_real_temp:
            # Order here should match order of original tensor... But what is the best way todo that...?
            # Hmm right now this will create a bug if for some reason the order [precip, temp, output]
            intial_numpy = (
                torch.stack(
                    [
                        output.view(-1).float().to(model.device),
                        precip_cols[i].float().to(model.device),
                        temp_cols[i].float().to(model.device),
                    ]
                )
                .to("cpu")
                .detach()
                .numpy()
            )
            temp_df = pd.DataFrame(intial_numpy.T, columns=rel_cols)
            revised_np = temp_df[rel_cols].to_numpy()
            full_history.append(
                torch.from_numpy(revised_np).to(model.device).unsqueeze(0)
            )
    remainder = forecast_length - hours_to_forecast % forecast_length
    if remainder != forecast_length:
        # Subtract remainder from array
        end_tensor = torch.cat(all_tensor, axis=0).to("cpu").detach()[:-remainder]
    else:
        end_tensor = torch.cat(all_tensor, axis=0).to("cpu").detach()

    print(end_tensor.shape)  # Dimension now is (n_time_steps_to_forecast_steps)!! i.e [16]
    return end_tensor


def generate_decoded_predictions(
    model: Type[TimeSeriesModel],
    test_data: CSVTestLoader,
    forecast_start_idx: int,
    device: torch.device,
    history_dim: torch.Tensor,
    hours_to_forecast: int,
    decoder_params: Dict,
    multi_targets: int = 1,
    targs: Union[bool, torch.Tensor] = False
) -> torch.Tensor:
    probabilistic = False
    scaler = None
    if test_data.no_scale:
        scaler = test_data
    if decoder_params is not None:
        if "probabilistic" in decoder_params:
            probabilistic = True

        real_target_tensor = (
            torch.from_numpy(test_data.df[forecast_start_idx:].to_numpy())
            .to(device)
            .unsqueeze(0)
            .to(model.device)
        )
        if targs:
            src = history_dim
            src0 = src[0]
            trg = targs
            decoder_seq_len = model.params["model_params"]["label_len"]
            end_tensor = decoding_function(model.model, src0, trg[1], model.params["dataset_params"]["forecast_length"],
                                           src[1], trg[0], 1, decoder_seq_len, hours_to_forecast, device)
        else:
            end_tensor = decoding_functions[decoder_params["decoder_function"]](
                model.model,
                history_dim,
                hours_to_forecast,
                real_target_tensor,
                decoder_params["unsqueeze_dim"],
                output_len=model.params["dataset_params"]["forecast_length"],
                multi_targets=multi_targets,
                device=model.device,
                probabilistic=probabilistic,
                scaler=scaler
            )
        if probabilistic:
            end_tensor_mean = end_tensor[0][:, :, 0].view(-1).to("cpu").detach()
            return end_tensor_mean, end_tensor[1]
        elif isinstance(end_tensor, tuple):
            e = end_tensor[0][:, :, 0].view(-1).to("cpu").detach(), end_tensor[1][:, :, 0].view(-1).to("cpu").detach()
            return e
        if multi_targets == 1:
            end_tensor = end_tensor[:, :, 0].view(-1)
    return end_tensor.to("cpu").detach()


def generate_prediction_samples(
    model: Type[TimeSeriesModel],
    df: pd.DataFrame,
    test_data: CSVTestLoader,
    history: torch.Tensor,
    device: torch.device,
    forecast_start_idx: int,
    forecast_length: int,
    hours_to_forecast: int,
    decoder_params: Dict,
    num_prediction_samples: int,
    multi_params=1,
    targs=False
) -> np.ndarray:
    """
    ss
    """
    pred_samples = []
    std_dev_samples = []
    probabilistic = False
    if decoder_params is not None:
        if "probabilistic" in decoder_params:
            probabilistic = True

    for _ in range(num_prediction_samples):
        end_tensor = generate_predictions(
            model,
            df,
            test_data,
            history,
            device,
            forecast_start_idx,
            forecast_length,
            hours_to_forecast,
            decoder_params,
            multi_params=multi_params,
            targs=targs
        )

        if probabilistic:
            pred_samples.append(end_tensor[0].numpy())
            std_dev_samples.append(end_tensor[1].numpy())
        else:
            pred_samples.append(end_tensor.numpy())
    if probabilistic:
        return np.array(pred_samples).T, np.array(std_dev_samples).T
    else:
        return np.array(pred_samples).T  # each column is 1 array of predictions
