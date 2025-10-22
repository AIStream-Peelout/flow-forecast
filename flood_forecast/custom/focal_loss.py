import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings
# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
) -> torch.Tensor:
    r"""
    Criterion that computes **Focal Loss** for multi-class classification.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
        - :math:`p_t` is the model's estimated probability for each class.

    :param input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
     :type input: torch.Tensor
     :param target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
     :type target: torch.Tensor
     :param alpha: Weighting factor :math:`\alpha \in [0, 1]`.
     :type alpha: float
     :param gamma: Focusing parameter :math:`\gamma >= 0`.
     :type gamma: float
     :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
     :type reduction: str
     :param eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.
     :type eps: Optional[float]
      :return: The computed Focal loss.
      :rtype: torch.Tensor
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    r"""
    Criterion that computes **Focal Loss** for multi-class classification.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
        - :math:`p_t` is the model's estimated probability for each class.

    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: Optional[float] = None) -> None:
        """
        Initializes the FocalLoss module.

        :param alpha: Weighting factor :math:`\alpha \in [0, 1]`.
         :type alpha: float
         :param gamma: Focusing parameter :math:`\gamma >= 0`.
         :type gamma: float
         :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
         :type reduction: str
         :param eps: Deprecated: scalar to enforce numerical stability. This is no longer used.
         :type eps: Optional[float]
          :return: None
          :rtype: None
        """
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Focal Loss.

        :param input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
         :type input: torch.Tensor
         :param target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
         :type target: torch.Tensor
          :return: The computed Focal loss.
          :rtype: torch.Tensor
        """
        if len(target.shape) == 3:
            target = target[:, 0, :]
        if len(target.shape) == 2:
            target = target[:, 0]
        if len(input.shape) == 3:
            input = input[:, 0, :]
        target = target.type(torch.int64)
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)


def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
) -> torch.Tensor:
    r"""
    Function that computes **Binary Focal Loss** from logits.

    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
        - :math:`p_t` is the model's estimated probability for each class.

    :param input: input data (logits) tensor of arbitrary shape.
     :type input: torch.Tensor
     :param target: the target tensor with shape matching input, containing values 0 or 1.
     :type target: torch.Tensor
     :param alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
     :type alpha: float
     :param gamma: Focusing parameter :math:`\gamma >= 0`.
     :type gamma: float
     :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
     :type reduction: str
     :param eps: Deprecated: scalar for numerically stability when dividing. This is no longer used.
     :type eps: Optional[float]
      :return: The computed Binary Focal loss.
      :rtype: torch.Tensor
    """

    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`binary_focal_loss_with_logits` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    probs_pos = torch.sigmoid(input)
    probs_neg = torch.sigmoid(-input)
    loss_tmp = -alpha * torch.pow(probs_neg, gamma) * target * F.logsigmoid(input) - (
        1 - alpha
    ) * torch.pow(probs_pos, gamma) * (1.0 - target) * F.logsigmoid(-input)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    r"""
    Criterion that computes **Binary Focal Loss** from logits.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
        - :math:`p_t` is the model's estimated probability for each class.

    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none') -> None:
        """
        Initializes the BinaryFocalLossWithLogits module.

        :param alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
         :type alpha: float
         :param gamma: Focusing parameter :math:`\gamma >= 0`.
         :type gamma: float
         :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
         :type reduction: str
          :return: None
          :rtype: None
        """
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Binary Focal Loss.

        :param input: input data (logits) tensor of arbitrary shape.
         :type input: torch.Tensor
         :param target: the target tensor with shape matching input, containing values 0 or 1.
         :type target: torch.Tensor
          :return: The computed Binary Focal loss.
          :rtype: torch.Tensor
        """
        return binary_focal_loss_with_logits(input, target, self.alpha, self.gamma, self.reduction)


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""
    Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    :param labels: tensor with integer labels of shape :math:`(N, *)`, where N is batch size.
     :type labels: torch.Tensor
     :param num_classes: number of classes in labels.
     :type num_classes: int
     :param device: the desired device of returned tensor.
     :type device: Optional[torch.device]
     :param dtype: the desired data type of returned tensor.
     :type dtype: Optional[torch.dtype]
     :param eps: A small scalar added for numerical stability to avoid zero probabilities.
     :type eps: float
      :return: The labels in one-hot tensor of shape :math:`(N, C, *)`.
      :rtype: torch.Tensor
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps