"""An implementation of entmax (Peters et al., 2019).

See https://arxiv.org/pdf/1905.05702 for detailed description. This builds on previous work with sparsemax (Martins &
Astudillo, 2016). See https://arxiv.org/pdf/1602.02068.
"""
# Author: Ben Peters
# Author: Vlad Niculae <vlad@vene.ro>
# License: MIT
# Code from https://github.com/deep-spin/entmax/blob/master/entmax/activations.py
# Their package seemed broken from some reason :(

import torch
import torch.nn as nn
from torch.autograd import Function


def _make_ix_like(X, dim):
    """
    Creates an index tensor in the shape of the input tensor for use in sorting operations.

    :param X: The input tensor.
    :type X: torch.Tensor
    :param dim: The dimension along which to create the index.
    :type dim: int
    :return: An index tensor with values [1, 2, ..., d] aligned to the specified dimension.
    :rtype: torch.Tensor
    """
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    """
    Rolls the specified dimension to the last axis of the tensor.

    :param X: The input tensor.
    :type X: torch.Tensor
    :param dim: The dimension to move to the last position.
    :type dim: int
    :return: Tensor with the specified dimension rolled to the last axis.
    :rtype: torch.Tensor
    """
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    """
    Computes the threshold and support size for sparsemax.

    :param X: The input tensor.
    :type X: torch.Tensor
    :param dim: The dimension along which to compute the threshold.
    :type dim: int
    :param k: Number of top elements to consider for partial sorting.
    :type k: int or None
    :return: A tuple containing the threshold and support size tensor.
    :rtype: tuple
    """
   

    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


def _entmax_threshold_and_support(X, dim=-1, k=None):
    """
    Computes the threshold and support size for 1.5-entmax.

    :param X: The input tensor.
    :type X: torch.Tensor
    :param dim: The dimension along which to compute the threshold.
    :type dim: int
    :param k: Number of top elements to consider for partial sorting.
    :type k: int or None
    :return: A tuple containing the threshold and support size tensor.
    :rtype: tuple
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star, support_size


class SparsemaxFunction(Function):
    """
    Autograd function for the sparsemax activation.
    Implements forward and backward passes.
    """
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        """
        Forward pass of sparsemax.

        :param ctx: Context to save tensors for backward pass.
        :param X: Input tensor.
        :type X: torch.Tensor
        :param dim: Dimension along which to apply sparsemax.
        :type dim: int
        :param k: Number of elements for partial sort.
        :type k: int or None
        :return: Sparsemax output tensor.
        :rtype: torch.Tensor
        """
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        """
        Backward pass of sparsemax.

        :param ctx: Context from forward pass.
        :param grad_output: Gradient of loss w.r.t output.
        :type grad_output: torch.Tensor
        :return: Gradient of loss w.r.t input.
        :rtype: tuple
        """
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


class Entmax15Function(Function):
    """
    Autograd function for the 1.5-entmax activation.
    Implements forward and backward passes.
    """
    @classmethod
    def forward(cls, ctx, X: torch.Tensor, dim=0, k=None):
        """
        Forward pass of 1.5-entmax.

        :param ctx: Context to save tensors for backward pass.
        :param X: Input tensor.
        :type X: torch.Tensor
        :param dim: Dimension along which to apply entmax15.
        :type dim: int
        :param k: Number of elements for partial sort.
        :type k: int or None
        :return: Output tensor after entmax15 transformation.
        :rtype: torch.Tensor
        """
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @classmethod
    def backward(cls, ctx, dY):
        """
        Backward pass of 1.5-entmax.

        :param ctx: Context from forward pass.
        :param dY: Gradient of loss w.r.t output.
        :type dY: torch.Tensor
        :return: Gradient of loss w.r.t input.
        :rtype: tuple
        """
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


def sparsemax(X, dim=-1, k=None):
    """
    Applies the sparsemax activation function along a given dimension.

    :param X: Input tensor.
    :type X: torch.Tensor
    :param dim: Dimension to apply sparsemax on.
    :type dim: int
    :param k: Number of top elements to consider for partial sorting.
    :type k: int or None
    :return: Tensor with sparsemax transformation applied.
    :rtype: torch.Tensor
    """

    return SparsemaxFunction.apply(X, dim, k)


def entmax15(X, dim=-1, k=None):
    """
    Applies the 1.5-entmax activation function along a given dimension.

    :param X: Input tensor.
    :type X: torch.Tensor
    :param dim: Dimension to apply entmax15 on.
    :type dim: int
    :param k: Number of top elements to consider for partial sorting.
    :type k: int or None
    :return: Tensor with entmax15 transformation applied.
    :rtype: torch.Tensor
    """

    return Entmax15Function.apply(X, dim, k)


class Sparsemax(nn.Module):
    """
    Torch module for the sparsemax activation function.

    :param dim: Dimension to apply sparsemax.
    :type dim: int
    :param k: Number of top elements to consider for partial sorting.
    :type k: int or None
    """
    def __init__(self, dim=-1, k=None):
        """
        Initializes the Sparsemax module.

        :param dim: Dimension along which to apply sparsemax.
        :type dim: int
        :param k: Number of top elements to consider for partial sorting. If None, full sort is used.
        :type k: int or None
        """
        self.dim = dim
        self.k = k
        super(Sparsemax, self).__init__()

    def forward(self, X):
        """
        Applies sparsemax to input tensor.

        :param X: Input tensor.
        :type X: torch.Tensor
        :return: Output tensor after sparsemax.
        :rtype: torch.Tensor
        """
        return sparsemax(X, dim=self.dim, k=self.k)


class Entmax15(nn.Module):
    """
    Torch module for the 1.5-entmax activation function.

    :param dim: Dimension to apply entmax15.
    :type dim: int
    :param k: Number of top elements to consider for partial sorting.
    :type k: int or None
    """
    def __init__(self, dim=-1, k=None):

        """
        Initializes the Entmax15 module.

        :param dim: Dimension along which to apply 1.5-entmax.
        :type dim: int
        :param k: Number of top elements to consider for partial sorting. If None, full sort is used.
        :type k: int or None
        """
        self.dim = dim
        self.k = k
        super(Entmax15, self).__init__()

    def forward(self, X: torch.Tensor):
        """
        Applies entmax15 to input tensor.

        :param X: Input tensor.
        :type X: torch.Tensor
        :return: Output tensor after entmax15.
        :rtype: torch.Tensor
        """
        return entmax15(X, dim=self.dim, k=self.k)
