from flood_forecast.transformer_xl.multi_head_base import MultiAttnHeadSimple
from flood_forecast.transformer_xl.transformer_basic import SimpleTransformer
from flood_forecast.transformer_xl.transformer_xl import TransformerXL
from torch.optim import Adam, SGD
from torch.nn import MSELoss, SmoothL1Loss, PoissonNLLLoss


pytorch_model_dict = {"MultiAttnHeadSimple":MultiAttnHeadSimple, "SimpleTransformer":SimpleTransformer, 
"TransformerXL":TransformerXL
}

pytorch_criterion_dict = {"MSE": MSELoss(), "SmoothL1Loss":SmoothL1Loss(), "PoissonNLLLoss":PoissonNLLLoss()}

pytorch_opt_dict = {"Adam":Adam, "SGD":SGD}

scikit_dict = {}