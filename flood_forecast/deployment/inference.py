from flood_forecast.time_model import PyTorchForecast
from flood_forecast.evaluator import infer_on_torch_model
from flood_forecast.plot_functions import plot_df_test_with_confidence_interval
from flood_forecast.explain_model_output import deep_explain_model_heatmap, deep_explain_model_summary_plot
from flood_forecast.time_model import scaling_function
# from flood_forecast.preprocessing.buil_dataset import get_data
from flood_forecast.gcp_integration.basic_utils import upload_file
from datetime import datetime
import pandas as pd
import wandb
# mport json


class InferenceMode(object):
    def __init__(self, hours_to_forecast: int, num_prediction_samples: int, model_params, csv_path: str, weight_path,
                 wandb_proj: str = None):
        """
        Class to handle inference for models.
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
        forecast_history = self.inference_params["dataset_params"]["forecast_history"]
        self.inference_params["datetime_start"] = some_date
        if csv_path:
            self.inference_params["test_csv_path"] = csv_path
            self.inference_params["dataset_params"]["file_path"] = csv_path
        df, tensor, history, forecast_start, test, samples = infer_on_torch_model(self.model, **self.inference_params)
        if test.scale:
            unscaled = test.inverse_scale(tensor.numpy().reshape(-1, 1))
            df["preds"][forecast_history:] = unscaled.numpy()[:, 0]
        if len(samples.columns) > 1:
            index = samples.index
            if hasattr(test, "targ_scaler"):
                samples = test.inverse_scale(samples)
                samples = pd.DataFrame(samples.numpy(), index=index)
            samples[:forecast_history] = 0
        if save_buck:
            df.to_csv("temp3.csv")
            upload_file(save_buck, save_name, "temp3.csv", self.model.gcs_client)
        return df, tensor, history, forecast_start, test, samples

    def make_plots(self, date: datetime, csv_path: str = None, csv_bucket: str = None,
                   save_name=None, wandb_plot_id=None):
        if csv_path is None:
            csv_path = self.csv_path
        df, tensor, history, forecast_start, test, samples = self.infer_now(date, csv_path, csv_bucket, save_name)
        plt = plot_df_test_with_confidence_interval(df, samples, forecast_start, self.model.params)
        if wandb_plot_id:
            wandb.log({wandb_plot_id: plt})
            deep_explain_model_summary_plot(self.model, test, date)
            deep_explain_model_heatmap(self.model, test, date)
        return tensor, history, test, plt


def load_model(model_params_dict, file_path, weight_path: str) -> PyTorchForecast:
    if weight_path:
        model_params_dict["weight_path"] = weight_path
    model_params_dict["inference_params"]["test_csv_path"] = file_path
    model_params_dict["inference_params"]["dataset_params"]["file_path"] = file_path
    m = PyTorchForecast(model_params_dict["model_name"], file_path, file_path, file_path, model_params_dict)
    return m
