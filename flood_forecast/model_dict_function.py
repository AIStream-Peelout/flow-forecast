from flood_forecast.transformer_xl.multi_head_base import MultiAttnHeadSimple
from flood_forecast.transformer_xl.transformer_basic import SimpleTransformer, CustomTransformerDecoder
from flood_forecast.transformer_xl.transformer_xl import TransformerXL
from flood_forecast.transformer_xl.dummy_torch import DummyTorchModel
from flood_forecast.basic.linear_regression import SimpleLinearModel
from flood_forecast.basic.lstm_vanilla import LSTMForecast
from torch.optim import Adam, SGD
from torch.nn import MSELoss, SmoothL1Loss, PoissonNLLLoss
from flood_forecast.custom.custom_opt import BertAdam
from flood_forecast.basic.linear_regression import simple_decode
from flood_forecast.transformer_xl.transformer_basic import greedy_decode
from flood_forecast.custom.custom_opt import RMSELoss, MAPELoss
# criterion_params
# { "quantile:""
#  }
import torch

"""
Utility dictionaries to map a string to a class
"""
pytorch_model_dict = {
    "MultiAttnHeadSimple": MultiAttnHeadSimple,
    "SimpleTransformer": SimpleTransformer,
    "TransformerXL": TransformerXL,
    "DummyTorchModel": DummyTorchModel,
    "LSTM": LSTMForecast,
    "SimpleLinearModel": SimpleLinearModel,
    "CustomTransformerDecoder": CustomTransformerDecoder}
pytorch_criterion_dict = {
    "MSE": MSELoss(),
    "SmoothL1Loss": SmoothL1Loss(),
    "PoissonNLLLoss": PoissonNLLLoss(),
    "RMSE": RMSELoss(),
    "MAPE": MAPELoss()}


evaluation_functions_dict = {"NSE": "", "MSE": ""}

decoding_functions = {"greedy_decode": greedy_decode, "simple_decode": simple_decode}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "BertAdam": BertAdam}

scikit_dict = {}


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
