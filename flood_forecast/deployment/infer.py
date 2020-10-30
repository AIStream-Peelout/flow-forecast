from flood_forecast.time_model import PyTorchForecast
from typing import Dict, List, Optional, Sequence, DefaultDict
from flood_forecast.evaluator import generate_prediction_samples


def load_model(model_params: Dict, new_csv_path: str, weight_path: str = None) -> PyTorchForecast:
    if weight_path:
        model_params["weight_path"] = weight_path
    model = PyTorchForecast(model_params["model_name"], new_csv_path, new_csv_path, new_csv_path, model_params)
    model.model.eval()
    return model


def run_inference_csv(model, csv_path: str = None, start_datetime=None):
    params = {}
    generate_prediction_samples(model, **params)


def new_instantiation(mode: str, model_params: DefaultDict, seq: Sequence, a_list: List) -> Optional:
    pass
