from flood_forecast.time_model import PyTorchForecast
from flood_forecast.evaluator import infer_on_torch_model
from flood_forecast.plot_functions import plot_df_test_with_confidence_interval
from flood_forecast.explain_model_output import deep_explain_model_heatmap, deep_explain_model_summary_plot
from torch.utils.data import DataLoader
from flood_forecast.time_model import scaling_function
# from flood_forecast.preprocessing.buil_dataset import get_data
from flood_forecast.gcp_integration.basic_utils import upload_file
from datetime import datetime
import wandb
import torch
from typing import Union, Dict, Tuple, List, Any
import pandas as pd


class InferenceMode(object):
    def __init__(self, forecast_steps: int, num_prediction_samples: int, model_params: Dict[str, Any],
                 csv_path: Union[str, pd.DataFrame], weight_path: str, wandb_proj: Union[str, None] = None, torch_script=False):
        """
        Class to handle inference for time series models, providing methods for forecasting and classification.

        :param forecast_steps: Number of time-steps to forecast (e.g., hours).
        :type forecast_steps: int
        :param num_prediction_samples: The number of prediction samples to generate (e.g., for confidence intervals).
        :type num_prediction_samples: int
        :param model_params: A dictionary of model parameters (ideally this should come from saved JSON config file).
        :type model_params: dict
        :param csv_path: Path to the CSV test file or a Pandas dataframe to be used for inference.
        :type csv_path: typing.Union[str, pandas.DataFrame]
        :param weight_path: Path to the model weights (.pth file).
        :type weight_path: str
        :param wandb_proj: The name of the W&B project; leave blank if you don't want to log to Wandb. Defaults to None.
        :type wandb_proj: typing.Union[str, None]
        :param torch_script: Whether the model should be loaded as a torch script model.
        :type torch_script: bool
        """
        if "inference_params" not in model_params:
            model_params["inference_params"] = {"dataset_params": {}}
        self.csv_path = csv_path
        self.hours_to_forecast = forecast_steps
        self.n_targets = model_params.get("n_targets")
        self.targ_cols = model_params["dataset_params"]["target_col"]
        self.model = load_model(model_params.copy(), csv_path, weight_path)
        self.inference_params = model_params["inference_params"]
        if "scaling" in self.inference_params["dataset_params"]:
            s = scaling_function({}, self.inference_params["dataset_params"])["scaling"]
            self.inference_params["dataset_params"]["scaling"] = s
        self.inference_params["hours_to_forecast"] = forecast_steps
        self.inference_params["num_prediction_samples"] = num_prediction_samples
        if wandb_proj:
            date = datetime.now()
            wandb.init(name=date.strftime("%H-%M-%D-%Y") + "_prod", project=wandb_proj)
            wandb.config.update(model_params, allow_val_change=True)

    def infer_now(self, some_date: datetime, csv_path: Union[str, None] = None, save_buck: Union[str, None] = None, save_name: Union[str, None] = None, use_torch_script: bool = False) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, int, Any, pd.DataFrame]:
        """
        Performs time series forecasting inference on a CSV file at a specified date-time.

        :param some_date: The date and time when the forecast should begin.
        :type some_date: datetime
        :param csv_path: An optional path to a CSV you want to perform inference on. Overrides the instance path. Defaults to None.
        :type csv_path: typing.Union[str, None]
        :param save_buck: The GCP bucket where you want to save the final predictions CSV. Defaults to None.
        :type save_buck: typing.Union[str, None]
        :param save_name: The name of the file to save the Pandas dataframe to GCP as. Defaults to None.
        :type save_name: typing.Union[str, None]
        :param use_torch_script: Optional parameter which allows you to use a saved torch script version of your model.
        :type use_torch_script: bool
        :return: A tuple containing the results: predictions dataframe, prediction tensor, historical tensor, 
                 forecast start index, test data loader, and the prediction samples dataframe (for CIs).
        :rtype: typing.Tuple[pandas.DataFrame, torch.Tensor, torch.Tensor, int, object, pandas.DataFrame]
        """
        forecast_history = self.inference_params["dataset_params"]["forecast_history"]
        self.inference_params["datetime_start"] = some_date
        if csv_path:
            self.inference_params["test_csv_path"] = csv_path
            self.inference_params["dataset_params"]["file_path"] = csv_path
        df, tensor, history, forecast_start, test, samples = infer_on_torch_model(self.model, **self.inference_params)
        print("the tensor shape is 2 ")
        print(tensor.shape)
        if test.scale and self.n_targets:
            unscaled = test.inverse_scale(tensor.numpy())
            for i in range(0, self.n_targets):
                df["pred_" + self.targ_cols[i]] = 0
                print("Shape of unscaled is: ")
                print(unscaled.shape)
                df["pred_" + self.targ_cols[i]][forecast_history:] = unscaled[:, i].numpy()
        elif test.scale:
            unscaled = test.inverse_scale(tensor.numpy().reshape(-1, 1))
            df["preds"][forecast_history:] = unscaled.numpy()[:, 0]
        if len(samples) > 0:
            for i in range(0, len(samples)):
                samples[i][:forecast_history] = 0
        if save_buck:
            df.to_csv("temp3.csv")
            upload_file(save_buck, save_name, "temp3.csv", self.model.gcs_client)
        return df, tensor, history, forecast_start, test, samples

    def infer_now_classification(self, data: Union[pd.DataFrame, str, None] = None, over_lap_seq: bool = True, save_buck: Union[str, None] = None, save_name: Union[str, None] = None, batch_size: int = 1) -> List[Any]:
        """
        Function to perform classification/anomaly detection on sequences in real-time.

        :param data: The data to perform inference on (DataFrame or file path). Defaults to None, using internal test data.
        :type data: typing.Union[pandas.DataFrame, str, None]
        :param over_lap_seq: Whether to slide the sequence window by one step (True) or by the full sequence length (False).
        :type over_lap_seq: bool
        :param save_buck: The GCP bucket where you want to save the results.
        :type save_buck: typing.Union[str, None]
        :param save_name: The name of the file to save the results to GCP as.
        :type save_name: typing.Union[str, None]
        :param batch_size: The batch size to use for inference.
        :type batch_size: int
        :return: A list of sequence predictions/outputs from the model.
        :rtype: typing.List[typing.Any]
        """
        if data is not None:
            dataset_params = self.model.params["dataset_params"].copy()
            dataset_params["class"] = "GeneralClassificationLoader"
            dataset_1 = self.model.make_data_load(data, dataset_params, "custom")
            inferL = DataLoader(dataset_1, batch_size=batch_size)
        else:
            loader = self.model.test_data
            inferL = DataLoader(loader, batch_size=batch_size)
        seq_list = []
        if over_lap_seq:
            for x, y in inferL:
                seq_list.append(self.model.model(x))
        else:
            for i in range(0, len(loader), dataset_params["sequence_length"]):
                loader[i]  # TODO finish implementing
        return seq_list

    def make_plots(self, date: datetime, csv_path: Union[str, None] = None, csv_bucket: Union[str, None] = None,
                   save_name: Union[str, None] = None, wandb_plot_id: Union[str, None] = None) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
        """
        Function to create forecast plots and optionally log them to Weights & Biases.

        :param date: The datetime to start the inference and plotting from.
        :type date: datetime
        :param csv_path: The path to the CSV file you want to use for inference. Defaults to None.
        :type csv_path: typing.Union[str, None]
        :param csv_bucket: The GCS bucket where the CSV file is located. Defaults to None.
        :type csv_bucket: typing.Union[str, None]
        :param save_name: The name to use when saving the output CSV to the bucket. Defaults to None.
        :type save_name: typing.Union[str, None]
        :param wandb_plot_id: The id to save the generated plot as on the Wandb dashboard. Defaults to None.
        :type wandb_plot_id: typing.Union[str, None]
        :return: A tuple containing the prediction tensor, history tensor, test data loader, and the last generated plot object.
        :rtype: typing.Tuple[torch.Tensor, torch.Tensor, object, object]
        """
        if csv_path is None:
            csv_path = self.csv_path
        df, tensor, history, forecast_start, test, samples = self.infer_now(date, csv_path, csv_bucket, save_name)
        plt = {}
        for sample, targ in zip(samples, self.model.params["dataset_params"]["target_col"]):
            plt = plot_df_test_with_confidence_interval(df, sample, forecast_start, self.model.params, targ)
            if wandb_plot_id:
                wandb.log({wandb_plot_id + targ: plt})
                if not self.n_targets:
                    deep_explain_model_summary_plot(self.model, test, date)
                    deep_explain_model_heatmap(self.model, test, date)
        return tensor, history, test, plt


def convert_to_torch_script(model: PyTorchForecast, save_path: str) -> PyTorchForecast:
    """
    Function to convert a PyTorch model to TorchScript using tracing and save the script.

    :param model: The PyTorchForecast model you wish to convert.
    :type model: flood_forecast.time_model.PyTorchForecast
    :param save_path: File path to save the TorchScript model under.
    :type save_path: str
    :return: Returns the original model instance with the TorchScript model added as a ``.script_model`` attribute.
    :rtype: flood_forecast.time_model.PyTorchForecast
    """
    model.model.eval()
    forecast_history = model.params["dataset_params"]["forecast_history"]
    n_features = model.params["model_params"]["n_time_series"]
    test_input = torch.rand(2, forecast_history, n_features)
    model_script = torch.jit.trace(model.model, test_input)
    test_input1 = torch.rand(4, forecast_history, n_features)
    a = model_script(test_input1)
    b = model.model(test_input1)
    model.script_model = model_script
    assert torch.eq(a, b).all()
    model_script.save(save_path)
    return model


def convert_to_onnx():
    """
    Converts a model to ONNX. (Function currently not implemented)

    :return: None
    :rtype: None
    """
    pass


def load_model(model_params_dict: Dict[str, Any], file_path: Union[str, pd.DataFrame], weight_path: str) -> PyTorchForecast:
    """
    Function to load a PyTorchForecast model from an existing configuration and specified weights.

    :param model_params_dict: Dictionary containing all model and dataset configuration parameters.
    :type model_params_dict: dict
    :param file_path: The path to the CSV or DataFrame used for initializing the data loader for inference.
    :type file_path: typing.Union[str, pandas.DataFrame]
    :param weight_path: The path to the model weights file (can be a local path or GCS path).
    :type weight_path: str
    :return: Returns a PyTorchForecast model initialized with the proper data and weights.
    :rtype: flood_forecast.time_model.PyTorchForecast
    """
    if weight_path:
        model_params_dict["weight_path"] = weight_path
    model_params_dict["inference_params"]["test_csv_path"] = file_path
    model_params_dict["inference_params"]["dataset_params"]["file_path"] = file_path
    if "weight_path_add" in model_params_dict:
        if "excluded_layers" in model_params_dict["weight_path_add"]:
            del model_params_dict["weight_path_add"]["excluded_layers"]
            # do stuff
    m = PyTorchForecast(model_params_dict["model_name"], file_path, file_path, file_path, model_params_dict)
    return m
