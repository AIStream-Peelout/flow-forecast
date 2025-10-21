import torch
from typing import Type


class SimpleLinearModel(torch.nn.Module):
    """
    A very simple baseline linear model to resolve some of the difficulties with bugs in the various train/validation
    loops in code.

    Has only two layers.

    :param seq_length:  Length of the input sequence
     :type seq_length: int
     :param n_time_series:  Number of time series channels
     :type n_time_series: int
     :param output_seq_len:  Number of output time steps to forecast
     :type output_seq_len: int
     :param probabilistic: Whether to output a probabilistic forecast
     :type probabilistic: bool
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
        Forward pass of the model.

        :param x: Input tensor of shape (B, L, M)
        :type x: torch.Tensor
        :return: Output tensor or Normal distribution if probabilistic
        :rtype: torch.Tensor or torch.distributions.Normal
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
    """
    Processes Gaussian output by averaging mean and standard deviation.

    :param out: Tuple containing two output tensors (mean, std)
    :type out: tuple
    :return: Averaged output, mean, and std tensors
    :rtype: tuple
    """
    out1 = torch.mean(torch.stack([out[0], out[1]]), dim=0)
    return out1, out[0], out[1]


def handle_no_scaling(scaler: torch.utils.data.Dataset, out: torch.Tensor, multi_targets: int):
    """
    Applies inverse transformation using scaler to the output.

    :param scaler: Dataset containing target scaler
    :type scaler: torch.utils.data.Dataset
    :param out: Output tensor to be transformed
    :type out: torch.Tensor
    :param multi_targets: Number of target series
    :type multi_targets: int
    :return: Transformed output
    :rtype: numpy.ndarray
    """
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
    Decodes output sequence from the model using a greedy-like autoregressive approach.

    :param model: PyTorch model to be used for decoding
    :type model: Type[torch.nn.Module]
    :param src: Source input tensor of shape (B, L, M)
    :type src: torch.Tensor
    :param max_seq_len: Maximum length of sequence to forecast
    :type max_seq_len: int
    :param real_target: Ground truth tensor used during decoding
    :type real_target: torch.Tensor
    :param start_symbol: Placeholder for API compatibility (not used)
    :type start_symbol: Any
    :param output_len: Length of model output per forward pass
    :type output_len: int
    :param device: Device used for model (default: 'cpu')
    :type device: str
    :param unsqueeze_dim: Dimension along which to unsqueeze input
    :type unsqueeze_dim: int
    :param meta_data: Optional metadata passed to the model
    :type meta_data: Any
    :param multi_targets: Number of target variables
    :type multi_targets: int
    :param use_real_target: Whether to use the real target during decoding (default: True)
    :type use_real_target: bool
    :param probabilistic: Whether the model outputs a probabilistic distribution
    :type probabilistic: bool
    :param scaler: Optional scaler for inverse transforming the output
    :type scaler: Any
    :return: Forecasted tensor(s) depending on model output mode (standard or probabilistic)
    :rtype: torch.Tensor or tuple
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
                    upper_out.append(up[:, :residual])
                    lower_out.append(lower[:, :residual])
                    handle_gauss = True
                elif probabilistic:
                    out_std = out.stddev.detach()
                    out = out.mean.detach()
                    ys_std_dev.append(out_std[:, 0:residual])
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
                if output_len != out.shape[1]:
                    raise ValueError("Output length should laways equal the output shape")
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
