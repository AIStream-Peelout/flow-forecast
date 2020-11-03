from flood_forecast.time_model import PyTorchForecast
from flood_forecast.evaluator import infer_on_torch_model
from flood_forecast.pre_dict import scaler_dict


class InferenceMode(object):
    def __init__(self, hours_to_forecast, num_prediction_samples, model_params, csv_path, weight_path):
        self.hours_to_forecast = hours_to_forecast
        self.model = load_model(model_params, weight_path, csv_path)
        self.inference_params = model_params["inference_params"]
        # TODO move scaling to actual class
        s = self.inference_params["dataset_params"]["scaler"]
        self.inference_params["dataset_params"]["scaling"] = scaler_dict[s]
        self.inference_params["hours_to_forecast"] = hours_to_forecast
        self.inference_params["num_prediction_samples"] = num_prediction_samples

    def infer_now(self, some_date, csv_path=None):
        self.inference_params["start_datetime"] = some_date
        if csv_path:
            self.inference_params["test_csv_path"] = csv_path
        infer_on_torch_model(self.model, **self.inference_params)


def load_model(model_params_dict, file_path, weight_path) -> PyTorchForecast:
    if weight_path:
        model_params_dict["weight_path"] = weight_path
    m = PyTorchForecast(model_params_dict["model_name"], file_path, file_path, file_path, model_params_dict)
    return m
