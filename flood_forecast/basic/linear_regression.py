import torch
from typing import Type


class SimpleLinearModel(torch.nn.Module):
    """
    A very simple baseline model to resolve some of the
    difficulties with bugs in the various train/validation loops
    in code. Has only two layers.
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


def handle_gaussian_loss(out: tuple):
    out1 = torch.mean(torch.stack([out[0], out[1]]), dim=0)
    return out1, out[0], out[1]


def handle_no_scaling(scaler: torch.utils.data.Dataset, out: torch.Tensor, multi_targets: int):
    if multi_targets == 1:
        out = out.detach().cpu().reshape(-1, 1)
    if len(out.shape) > 2:
        out = out[0, :, :]
    out = scaler.targ_scaler.transform(out.detach().cpu())
    return out


def simple_decode(model: Type[torch.nn.Module],
                  src: torch.Tensor,
                  max_seq_len: int,
                  real_target: torch.Tensor,
                  start_symbol=None,
                  output_len=1,
                  device='cpu',
                  unsqueeze_dim=1,
                  meta_data=None,
                  multi_targets=1,
                  use_real_target: bool = True,
                  probabilistic: bool = False,
                  scaler=None) -> torch.Tensor:
    """
    :model a PyTorch model to be used for decoding
    :src the source tensor
    :the max length sequence to return
    :real_target the actual target values we want to forecast (don't worry they are masked)
    :start_symbol used to match the function signature of greedy_decode not ever used here though.
    :output_len potentially used to forecast multiple steps at once. Not implemented yet though.
    :device used to to match function signature
    :multi_task int: Multitask return will always be (batch_size, output_len, multi_targets)
    :returns a torch.Tensor of dimension (B, max_seq_len, M)
    """
    real_target = real_target.float()
    real_target2 = real_target.clone()
    # Use last value
    ys = src[:, -1, :].unsqueeze(unsqueeze_dim)
    ys_std_dev = []
    upper_out = []
    lower_out = []
    handle_gauss = False
    for i in range(0, max_seq_len, output_len):
        residual = output_len if max_seq_len - output_len - i >= 0 else max_seq_len % output_len
        with torch.no_grad():
            if meta_data:
                out = model(src, meta_data).unsqueeze(2)
            else:
                out = model(src)
                if isinstance(out, tuple):
                    out, up, lower = handle_gaussian_loss(out)
                    print(up)
                    upper_out.append(up[:, :residual])
                    lower_out.append(lower[:, :residual])
                    handle_gauss = True
                elif probabilistic:
                    out_std = out.stddev.detach()
                    out = out.mean.detach()
                    ys_std_dev.append(out_std[:, 0].unsqueeze(0))
                elif multi_targets < 2:
                    out = out.unsqueeze(2)
            if scaler:
                handle_no_scaling(scaler, out, multi_targets)
            if not isinstance(out, torch.Tensor):
                out = torch.from_numpy(out)
            if output_len == 1:
                real_target2[:, i, 0:multi_targets] = out[:, 0]
                src = torch.cat((src[:, 1:, :], real_target2[:, i, :].unsqueeze(1)), 1)
                ys = torch.cat((ys, real_target2[:, i, :].unsqueeze(1)), 1)
            else:
                # residual = output_len if max_seq_len - output_len - i >= 0 else max_seq_len % output_len
                real_target2[:, i:i + residual, 0:multi_targets] = out[:, :residual]
                src = torch.cat((src[:, residual:, :], real_target2[:, i:i + residual, :]), 1)
                ys = torch.cat((ys, real_target2[:, i:i + residual, :]), 1)
    if probabilistic:
        ys_std_dev = torch.cat(ys_std_dev, dim=1)
        return ys[:, 1:, 0:multi_targets], ys_std_dev
    if handle_gauss:
        return torch.cat(upper_out, dim=1), torch.cat(lower_out, dim=1), ys[:, 1:, 0:multi_targets]
    else:
        return ys[:, 1:, 0:multi_targets]
