import torch
from typing import Type


# TODO move decode example
class SimpleLinearModel(torch.nn.Module):
    """
    A very simple baseline model to resolve some of the
    difficulties with bugs in the various train/validation loops
    in  code. Has only two layers.
    """

    def __init__(self, seq_length: int, n_time_series: int, output_seq_len=1, probabilistic: bool = False):
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.initial_layer = torch.nn.Linear(n_time_series, 1)
        self.probabilistic = probabilistic
        if self.probabilistic:
            self.output_len = 2
        else:
            self.output_len = output_seq_len
        self.output_layer = torch.nn.Linear(seq_length, self.output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: A tensor of dimension (B, L, M) where
        B is the batch size, L is the length of the sequence
        """
        x = self.initial_layer(x)
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        if self.probabilistic:
            mean = x[..., 0][..., None]
            std = torch.clamp(x[..., 1][..., None], min=0.01)
            return torch.distributions.Normal(mean, std)
        else:
            return x.view(-1, self.output_len)


def simple_decode(model: Type[torch.nn.Module],
                  src: torch.Tensor,
                  max_seq_len: int,
                  real_target: torch.Tensor,
                  start_symbol=None,
                  output_len=1,
                  device='cpu',
                  unsqueeze_dim=1,
                  meta_data=None,
                  use_real_target: bool = True,
                  probabilistic: bool = False) -> torch.Tensor:
    """
    :model a PyTorch model to be used for decoding
    :src the source tensor
    :the max length sequence to return
    :real_target the actual target values we want to forecast (don't worry they are masked)
    :start_symbol used to match the function signature of greedy_decode not ever used here though.
    :output_len potentially used to forecast multiple steps at once. Not implemented yet though.
    :device used to to match function signature
    :returns a torch.Tensor of dimension (B, max_seq_len, M)
    """
    real_target = real_target.float()
    real_target2 = real_target.clone()
    # Use last value
    ys = src[:, -1, :].unsqueeze(unsqueeze_dim)
    ys_std_dev = []
    for i in range(0, max_seq_len, output_len):
        with torch.no_grad():
            if meta_data:
                out = model(src, meta_data)
            else:
                out = model(src)

            if probabilistic:
                out_std = out.stddev.detach()
                out = out.mean.detach()
                ys_std_dev.append(out_std[:, 0].unsqueeze(0))

            if output_len == 1:
                real_target2[:, i, 0] = out[:, 0]
                src = torch.cat((src[:, 1:, :], real_target2[:, i, :].unsqueeze(1)), 1)
                ys = torch.cat((ys, real_target2[:, i, :].unsqueeze(1)), 1)
            else:
                residual = output_len if max_seq_len - output_len - i >= 0 else max_seq_len % output_len
                real_target2[:, i:i + residual, 0] = out[:, :residual]
                src = torch.cat((src[:, residual:, :], real_target2[:, i:i + residual, :]), 1)
                ys = torch.cat((ys, real_target2[:, i:i + residual, :]), 1)
    if probabilistic:
        ys_std_dev = torch.cat(ys_std_dev, dim=1)
        return ys[:, 1:, :], ys_std_dev
    else:
        return ys[:, 1:, :]
