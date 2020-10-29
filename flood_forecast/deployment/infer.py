from flood_forecast.time_model import PyTorchForecast
from typing import Dict, List, Optional, Sequence, DefaultDict


def load_model(model_params: Dict, new_csv_path: str, weight_path: str = ""):
    if weight_path not in model_params:
        model_params["weight_path"] = weight_path
    model = PyTorchForecast(model_params["name"], new_csv_path, new_csv_path, new_csv_path, model_params)
    return model

def new():
    pass