import torch
class SimpleLinear(torch.nn.Module):
    def __init__(self, forecast_history:int, n_time_series:int):
        super.__init__()
        self.forecast_history = forecast_history
        self.n_time_series = n_time_series

    def forward(self, x:torch.Tensor):
        pass 
