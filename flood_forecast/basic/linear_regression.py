import torch
class SimpleLinearModel(torch.nn.Module):
    """
    A very simple baseline model to resolve some of the
    difficulties with bugs in the various train/validation loops 
    in code.
    """
    def __init__(self, seq_length:int, n_time_series:int, output_seq_len=1):
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.initial_layer = torch.nn.Linear(n_time_series, 1)
        self.output_layer = torch.nn.Linear(seq_length, output_seq_len)
        self.output_len = output_seq_len

    def forward(self, x: torch.Tensor):
        """
        x: A tensor of dimension (B, L, M) where 
        B is the batch size, L is the length of the 
        """
        x = self.initial_layer(x)
        x = x.permute(0,2,1)
        x = self.output_layer(x)
        return x.view(-1, self.output_len)

def simple_decode(model, src, max_seq_len, real_target, output_len, unsqueeze_dim=1):
    real_target2 = real_target.clone()
    ys = src[:, -1, :].unsqueeze(unsqueeze_dim)
    for i in range(0, max_seq_len):
        with torch.no_grad():
            out = model(src)
            real_target2[:, i, 0] = out[:, 0]
            src = torch.cat((src[:, 1:, :], real_target2[:, i, :].unsqueeze(1)), 1)
            ys = torch.cat((ys, real_target2[:, i, :].unsqueeze(1)), 1 )
    return ys[:, 1:, :]




