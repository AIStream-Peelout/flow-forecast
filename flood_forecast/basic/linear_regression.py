import torch
class SimpleLinear(torch.nn.Module):
    """
    A very simple baseline model to resolve some of the
    difficulties with bugs in the various train/validation loops 
    in code.
    """
    def __init__(self, forecast_history:int, n_time_series:int):
        super.__init__()
        self.forecast_history = forecast_history
        self.n_time_series = n_time_series
        self.initial_layer = torch.nn.Linear(n_time_series, 1)
        self.output_layer = torch.nn.Linear(forecast_history, 1)

    def forward(self, x:torch.Tensor):
        x = self.initial_layer(x)
        x = x.permute(0,2,1)
        x= self.output_layer(x)
        return x 
