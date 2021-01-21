from flood_forecast.time_model import PyTorchForecast
from flood_forecast.evaluator import infer_on_torch_model
from flood_forecast.plot_functions import plot_df_test_with_confidence_interval
from flood_forecast.explain_model_output import deep_explain_model_heatmap, deep_explain_model_summary_plot
from flood_forecast.time_model import scaling_function
# from flood_forecast.preprocessing.buil_dataset import get_data
from flood_forecast.gcp_integration.basic_utils import upload_file
from datetime import datetime
import wandb
# mport json


class InferenceMode(object):
    def __init__(self, hours_to_forecast: int, num_prediction_samples: int, model_params, csv_path: str, weight_path,
                 wandb_proj: str = None, torch_script=False):
        """Class to handle inference for models

        :param hours_to_forecast: [description]
        :type hours_to_forecast: int
        :param num_prediction_samples: [description]
        :type num_prediction_samples: int
        :param model_params: [description]
        :type model_params: [type]
        :param csv_path: [description]
        :type csv_path: str
        :param weight_path: [description]
        :type weight_path: [type]
        :param wandb_proj: [description], defaults to None
        :type wandb_proj: str, optional
        """
        self.hours_to_forecast = hours_to_forecast
        self.csv_path = csv_path
        self.model = load_model(model_params.copy(), csv_path, weight_path)
        self.inference_params = model_params["inference_params"]
        if "scaling" in self.inference_params["dataset_params"]:
            s = scaling_function({}, self.inference_params["dataset_params"])["scaling"]
            self.inference_params["dataset_params"]["scaling"] = s
        self.inference_params["hours_to_forecast"] = hours_to_forecast
        self.inference_params["num_prediction_samples"] = num_prediction_samples
        if wandb_proj:
            date = datetime.now()
            wandb.init(name=date.strftime("%H-%M-%D-%Y") + "_prod", project=wandb_proj)
            wandb.config.update(model_params)

    def infer_now(self, some_date, csv_path=None, save_buck=None, save_name=None):
        """[summary]

        :param some_date: [description]
        :type some_date: [type]
        :param csv_path: [description], defaults to None
        :type csv_path: [type], optional
        :param save_buck: [description], defaults to None
        :type save_buck: [type], optional
        :param save_name: [description], defaults to None
        :type save_name: [type], optional
        :return: [description]
        :rtype: [type]
        """
        forecast_history = self.inference_params["dataset_params"]["forecast_history"]
        self.inference_params["datetime_start"] = some_date
        if csv_path:
            self.inference_params["test_csv_path"] = csv_path
            self.inference_params["dataset_params"]["file_path"] = csv_path
        df, tensor, history, forecast_start, test, samples = infer_on_torch_model(self.model, **self.inference_params)
        if test.scale:
            unscaled = test.inverse_scale(tensor.numpy().reshape(-1, 1))
            df["preds"][forecast_history:] = unscaled.numpy()[:, 0]
        if len(samples) > 1:
            samples[:forecast_history] = 0
        if save_buck:
            df.to_csv("temp3.csv")
            upload_file(save_buck, save_name, "temp3.csv", self.model.gcs_client)
        return df, tensor, history, forecast_start, test, samples

    def make_plots(self, date: datetime, csv_path: str = None, csv_bucket: str = None,
                   save_name=None, wandb_plot_id=None):
        """[summary]

        :param date: [description]
        :type date: datetime
        :param csv_path: [description], defaults to None
        :type csv_path: str, optional
        :param csv_bucket: [description], defaults to None
        :type csv_bucket: str, optional
        :param save_name: [description], defaults to None
        :type save_name: [type], optional
        :param wandb_plot_id: [description], defaults to None
        :type wandb_plot_id: [type], optional
        :return: [description]
        :rtype: [type]
        """
        if csv_path is None:
            csv_path = self.csv_path
        df, tensor, history, forecast_start, test, samples = self.infer_now(date, csv_path, csv_bucket, save_name)
        plt = {}
        for sample in samples:
            plt = plot_df_test_with_confidence_interval(df, sample, forecast_start, self.model.params)
            if wandb_plot_id:
                wandb.log({wandb_plot_id: plt})
                deep_explain_model_summary_plot(self.model, test, date)
                deep_explain_model_heatmap(self.model, test, date)
        return tensor, history, test, plt


def load_model(model_params_dict, file_path, weight_path: str) -> PyTorchForecast:
    """[summary]

    :param model_params_dict: [description]
    :type model_params_dict: [type]
    :param file_path: [description]
    :type file_path: [type]
    :param weight_path: [description]
    :type weight_path: str
    :return: [description]
    :rtype: PyTorchForecast
    """
    if weight_path:
        model_params_dict["weight_path"] = weight_path
    model_params_dict["inference_params"]["test_csv_path"] = file_path
    model_params_dict["inference_params"]["dataset_params"]["file_path"] = file_path
    m = PyTorchForecast(model_params_dict["model_name"], file_path, file_path, file_path, model_params_dict)
    return m
