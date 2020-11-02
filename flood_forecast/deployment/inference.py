from flood_forecast.time_model import PyTorchForecast
from flood_forecast.evaluator import infer_on_torch_model


class InferenceMode(object):
    def __init__(self, hours_to_forecast, num_prediction_samples, model_params, csv_path, weight_path):
        self.hours_to_forecast = hours_to_forecast
        self.model = load_model(model_params, csv_path, weight_path)
        self.inference_params = model_params["inference_params"]
        self.inference_params["hours_to_forecast"] = hours_to_forecast
        self.inference_params["num_prediction_samples"] = num_prediction_samples

    def infer_now(self, some_date, csv_path=None, save_csv=None):
        self.inference_params["datetime_start"] = some_date
        if csv_path:
            self.inference_params["test_csv_path"] = csv_path
        df, tensor, history, forecast_start, test, samples = infer_on_torch_model(self.model, **self.inference_params)
        if save_csv:
            pass
        return df, tensor, history, forecast_start, test, samples

    def make_plots(self):
        pass


def load_model(model_params_dict, file_path, weight_path) -> PyTorchForecast:
    if weight_path:
        model_params_dict["weight_path"] = weight_path
    model_params_dict["inference_params"]["test_csv_path"] = file_path
    m = PyTorchForecast(model_params_dict["model_name"], file_path, file_path, file_path, model_params_dict)
    return m
