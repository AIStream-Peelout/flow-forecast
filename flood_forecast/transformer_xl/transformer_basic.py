from torch import nn
from flood_forecast.time_model import TimeSeriesModel
class TransformerModel(TimeSeriesTorch):
   def __init__(self, init_params, d_model, n_heads=8, layers=6):
      super().__init__(init_params)
   

