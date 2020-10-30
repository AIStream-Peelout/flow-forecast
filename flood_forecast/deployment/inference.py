from flood_forecast.time_model import PyTorchForecast


def load_model(model_params_dict, weight_path=None, file_path: str = "") -> PyTorchForecast:
    if weight_path:
        model_params_dict["weight_path"] = weight_path
    m = PyTorchForecast(model_params_dict, file_path, file_path, file_path, model_params_dict)
    return m
