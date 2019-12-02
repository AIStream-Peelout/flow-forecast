from flood_forecast.transformer_xl.multi_head_base import MultiAttnHeadSimple
from flood_forecast.transformer_xl.transformer_basic import SimpleTransformer
from flood_forecast.transformer_xl.transformer_xl import TransformerXL

pytorch_model_dict = {"MultiAttnHeadSimple":MultiAttnHeadSimple, "SimpleTransformer":SimpleTransformer, 
"TransformerXL":TransformerXL
}
pytorch_criterion_dict = {}

scikit_dict = {}