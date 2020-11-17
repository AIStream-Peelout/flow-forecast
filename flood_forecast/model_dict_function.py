from flood_forecast.transformer_xl.multi_head_base import MultiAttnHeadSimple
from flood_forecast.transformer_xl.transformer_basic import SimpleTransformer, CustomTransformerDecoder
from flood_forecast.transformer_xl.transformer_xl import TransformerXL
from flood_forecast.transformer_xl.dummy_torch import DummyTorchModel
from flood_forecast.basic.linear_regression import SimpleLinearModel
from flood_forecast.basic.lstm_vanilla import LSTMForecast
from torch.optim import Adam, SGD
from torch.nn import MSELoss, SmoothL1Loss, PoissonNLLLoss, L1Loss
from flood_forecast.custom.custom_opt import BertAdam
from flood_forecast.basic.linear_regression import simple_decode
from flood_forecast.transformer_xl.transformer_basic import greedy_decode
from flood_forecast.da_rnn.model import DARNN
from flood_forecast.custom.custom_opt import RMSELoss, MAPELoss, PenalizedMSELoss, NegativeLogLikelihood, MASELoss
from flood_forecast.transformer_xl.transformer_bottleneck import DecoderTransformer
from flood_forecast.custom.dilate_loss import DilateLoss
from flood_forecast.meta_models.basic_ae import AE
import torch

"""
Utility dictionaries to map a string to a class.
"""
pytorch_model_dict = {
    "MultiAttnHeadSimple": MultiAttnHeadSimple,
    "SimpleTransformer": SimpleTransformer,
    "TransformerXL": TransformerXL,
    "DummyTorchModel": DummyTorchModel,
    "LSTM": LSTMForecast,
    "SimpleLinearModel": SimpleLinearModel,
    "CustomTransformerDecoder": CustomTransformerDecoder,
    "DARNN": DARNN,
    "DecoderTransformer": DecoderTransformer,
    "BasicAE": AE
}

pytorch_criterion_dict = {
    "MASELoss": MASELoss,
    "MSE": MSELoss,
    "SmoothL1Loss": SmoothL1Loss,
    "PoissonNLLLoss": PoissonNLLLoss,
    "RMSE": RMSELoss,
    "MAPE": MAPELoss,
    "DilateLoss": DilateLoss,
    "L1": L1Loss,
    "PenalizedMSELoss": PenalizedMSELoss,
    "NegativeLogLikelihood": NegativeLogLikelihood}

decoding_functions = {"greedy_decode": greedy_decode, "simple_decode": simple_decode}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "BertAdam": BertAdam}


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
