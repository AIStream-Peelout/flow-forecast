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
from typing import Union
import pandas as pd


class InferenceMode(object):
    def __init__(self, forecast_steps: int, n_samp: int, model_params, csv_path: Union[str, pd.DataFrame], weight_path,
                 wandb_proj: str = None, torch_script=False):
        """Class to handle inference for models,

        :param forecasts_steps: Number of time-steps to forecast (doesn't have to be hours)
        :type forecast_steps: int
        :param num_prediction_samples: Number of prediction samples
        :type num_prediction_samples: int
        :param model_params: A dictionary of model parameters (ideally this should come from saved JSON config file)
        :type model_params: Dict
        :param csv_path: Path to the CSV test file you want to be used for inference or a Pandas dataframe.
        :type csv_path: str
        :param weight_path: Path to the model weights (.pth file)
        :type weight_path: str
        :param wandb_proj: The name of the WB project leave blank if you don't want to log to Wandb, defaults to None
        :type wandb_proj: str, optionals
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
        self.inference_params["num_prediction_samples"] = n_samp
        if wandb_proj:
            date = datetime.now()
            wandb.init(name=date.strftime("%H-%M-%D-%Y") + "_prod", project=wandb_proj)
            wandb.config.update(model_params, allow_val_change=True)

    def infer_now(self, some_date: datetime, csv_path=None, save_buck=None, save_name=None, use_torch_script=False):
        """Performs inference on a CSV file at a specified date-time

        :param some_date: The date you want inference to begin on.
        :param csv_path: A path to a CSV you want to perform inference on, defaults to None
        :type csv_path: str, optional
        :param save_buck: The GCP bucket where you want to save predictions, defaults to None
        :type save_buck: str, optional
        :param save_name: The name of the file to save the Pandas data-frame to GCP as, defaults to None
        :type save_name: str, optional
        :param use_torch_script: Optional parameter which allows you to use a saved torch script version of your model.
        :return: Returns a tuple consisting of the Pandas dataframe with predictions + history,
        the prediction tensor, a tensor of the historical values, the forecast start index, the test loader, and the
        a dataframe of the prediction samples (e.g. the confidence interval preds)
        :rtype: tuple(pd.DataFrame, torch.Tensor, int, CSVTestLoader, pd.DataFrame)
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

    def infer_now_classification(self, data=None, over_lap_seq=True, save_buck=None, save_name=None, batch_size=1):
        """Function to preform classification/anomaly detection on sequences in real-time
        :param data
        :type data: Union[pd.DataFrame, str], optional
        :param over_lap_seq: Whether to increment by one throughout the df or by sequence length
        :type over_lap_seq: bool,
        :param batch_size: The batch size to use, defaults to 1

        """
        if data:
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

    def make_plots(self, date: datetime, csv_path: str = None, csv_bucket: str = None,
                   save_name=None, wandb_plot_id=None):
        """Function to create plots in inference mode.

        :param date: The datetime to start inference
        :type date: datetime
        :param csv_path: The path to the CSV file or  you want to use for inference, defaults to None
        :type csv_path: str, optional
        :param csv_bucket: The bucket where the CSV file is located, defaults to None
        :type csv_bucket: str, optional
        :param save_name: Where to save the output csv, defaults to None
        :type save_name: str, optional
        :param wandb_plot_id: The id to save wandb plot as on dashboard, defaults to None
        :type wandb_plot_id: str, optional
        :return: [description]
        :rtype: tuple(torch.Tensor, torch.Tensor, CSVTestLoader, matplotlib.pyplot.plot)
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
    """Function to convert PyTorch model to torch script and save

    :param model: The PyTorchForecast model you wish to convert
    :type model: PyTorchForecast
    :param save_path: File name to save the TorchScript model under.
    :type save_path: str
    :return: Returns the model with an added .script_model attribute
    :rtype: PyTorchForecast
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
    """"""
    pass


def load_model(model_params_dict, file_path: str, weight_path: str) -> PyTorchForecast:
    """Function to load a PyTorchForecast model from an existing config file.

    :param model_params_dict: Dictionary of model parameters
    :type model_params_dict: Dict
    :param file_path: The path to the CSV for running infer
    :type file_path: str
    :param weight_path: The path to the model weights (can be GCS)
    :type weight_path: str
    :return: Returns a PyTorchForecast model initialized with the proper data
    :rtype: PyTorchForecast
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
