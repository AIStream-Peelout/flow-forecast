import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging
from typing import List

import torch.distributions as tdist

# BERTAdam see
# https://github.com/huggingface/transformers/blob/694e2117f33d752ae89542e70b84533c52cb9142/pytorch_pretrained_bert/optimization.py

logger = logging.getLogger(__name__)


def warmup_cosine(x, warmup=0.002):
    """
    Cosine-based warmup learning rate schedule.

    :param x: Progress ratio (usually step / total_steps).
    :type x: float
    :param warmup: Proportion of steps used for warmup.
    :type warmup: float
    :return: Learning rate multiplier.
    :rtype: float
    """

    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    """
    Constant learning rate schedule with linear warmup.

    :param x: Progress ratio (usually step / total_steps).
    :type x: float
    :param warmup: Proportion of steps used for warmup.
    :type warmup: float
    :return: Learning rate multiplier.
    :rtype: float
    """

    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    """
    Linear learning rate schedule with warmup and linear decay to 0.

    :param x: Progress ratio (usually step / total_steps).
    :type x: float
    :param warmup: Proportion of steps used for warmup.
    :type warmup: float
    :return: Learning rate multiplier.
    :rtype: float
    """

    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


class MASELoss(torch.nn.Module):
    def __init__(self, baseline_method):
        """
        Initializes MASELoss with a baseline method.

        :param baseline_method: Baseline method to compare against (e.g. "mean").
        :type baseline_method: str
        """

        super(MASELoss, self).__init__()
        self.method_dict = {"mean": lambda x, y: torch.mean(x, 1).unsqueeze(1).repeat(1, y[1], 1)}
        self.baseline_method = self.method_dict[baseline_method]

    def forward(self, target: torch.Tensor, output: torch.Tensor, train_data: torch.Tensor, m=1) -> torch.Tensor:
        """
        Computes the Mean Absolute Scaled Error (MASE) loss.

        :param target: Ground truth tensor.
        :type target: torch.Tensor
        :param output: Model output predictions.
        :type output: torch.Tensor
        :param train_data: Historical training data used for naive baseline.
        :type train_data: torch.Tensor
        :param m: Seasonal periodicity (default is 1).
        :type m: int
        :return: MASE loss value.
        :rtype: torch.Tensor
        """

        # Ugh why can't all tensors have batch size... Fixes for modern
        if len(train_data.shape) < 3:
            train_data = train_data.unsqueeze(0)
        if m == 1 and len(target.shape) == 1:
            output = output.unsqueeze(0)
            output = output.unsqueeze(2)
            target = target.unsqueeze(0)
            target = target.unsqueeze(2)
        if len(target.shape) == 2:
            output = output.unsqueeze(0)
            target = target.unsqueeze(0)
        result_baseline = self.baseline_method(train_data, output.shape)
        MAE = torch.nn.L1Loss()
        mae2 = MAE(output, target)
        mase4 = MAE(result_baseline, target)
        # Prevent divison by zero/loss exploding
        if mase4 < 0.001:
            mase4 = 0.001
        return mae2 / mase4


class RMSELoss(torch.nn.Module):
    """Returns RMSE using:

    target -> True y
    output -> Prediction by model
    source: https://discuss.pytorch.org/t/rmse-loss-function/16540/3
    """

    def __init__(self, variance_penalty=0.0):
        """
        Initializes RMSELoss with optional variance penalty.

        :param variance_penalty: Coefficient for standard deviation penalty.
        :type variance_penalty: float
        """

        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.variance_penalty = variance_penalty

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        Computes the Root Mean Squared Error (RMSE), with optional variance penalty.

        :param output: Model output predictions.
        :type output: torch.Tensor
        :param target: Ground truth tensor.
        :type target: torch.Tensor
        :return: RMSE loss (plus optional variance penalty).
        :rtype: torch.Tensor
        """

        if len(output) > 1:

            diff = torch.sub(target, output)
            std_dev = torch.std(diff)
            var_penalty = self.variance_penalty * std_dev

            # torch.abs(target - output))
            print('diff', diff)
            print('std_dev', std_dev)
            print('var_penalty', var_penalty)
            return torch.sqrt(self.mse(target, output)) + var_penalty
        else:
            return torch.sqrt(self.mse(target, output))


class MAPELoss(torch.nn.Module):
    """Returns MAPE using:

    target -> True y output -> Predtion by model
    """

    def __init__(self, variance_penalty=0.0):
        """
        Initializes MAPELoss with optional variance penalty.

        :param variance_penalty: Coefficient for standard deviation penalty.
        :type variance_penalty: float
        """

        super().__init__()
        self.variance_penalty = variance_penalty

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        Computes the Mean Absolute Percentage Error (MAPE), with optional variance penalty.

        :param output: Model output predictions.
        :type output: torch.Tensor
        :param target: Ground truth values.
        :type target: torch.Tensor
        :return: MAPE loss (plus optional variance penalty).
        :rtype: torch.Tensor
        """

        if len(output) > 1:
            return torch.mean(torch.abs(torch.sub(target, output) / target)) + \
                self.variance_penalty * torch.std(torch.sub(target, output))
        else:
            return torch.mean(torch.abs(torch.sub(target, output) / target))


class PenalizedMSELoss(torch.nn.Module):
    """Returns MSE using:

    target -> True y
    output -> Predtion by model
    source: https://discuss.pytorch.org/t/rmse-loss-function/16540/3
    """

    def __init__(self, variance_penalty=0.0):
        """
        Initializes PenalizedMSELoss with optional variance penalty.

        :param variance_penalty: Coefficient for standard deviation penalty.
        :type variance_penalty: float
        """

        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.variance_penalty = variance_penalty

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        Computes the Mean Squared Error (MSE), with optional variance penalty.

        :param output: Model output predictions.
        :type output: torch.Tensor
        :param target: Ground truth tensor.
        :type target: torch.Tensor
        :return: Penalized MSE loss.
        :rtype: torch.Tensor
        """

        return self.mse(target, output) + \
            self.variance_penalty * torch.std(torch.sub(target, output))


# Add custom loss function
class GaussianLoss(torch.nn.Module):
    def __init__(self, mu=0, sigma=0):
        """
        Initializes GaussianLoss with fixed distribution parameters.

        :param mu: Mean of the Gaussian distribution.
        :type mu: float
        :param sigma: Standard deviation of the Gaussian distribution.
        :type sigma: float
        """

        super(GaussianLoss, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        """
        Computes negative log-likelihood for a Gaussian distribution.

        :param x: Input tensor (samples).
        :type x: torch.Tensor
        :return: Average negative log-likelihood over batch.
        :rtype: torch.Tensor
        """

        loss = - tdist.Normal(self.mu, self.sigma).log_prob(x)
        return torch.sum(loss) / (loss.size(0) * loss.size(1))


class QuantileLoss(torch.nn.Module):
    """From https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629."""

    def __init__(self, quantiles):
        """
        Initializes QuantileLoss with a list of quantiles.

        :param quantiles: List of quantiles to use for loss computation.
        :type quantiles: list or torch.Tensor
        """

        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        Computes the quantile loss across multiple quantiles.

        :param preds: Predicted quantile values, shape (batch_size, num_quantiles).
        :type preds: torch.Tensor
        :param target: Ground truth values, shape (batch_size,).
        :type target: torch.Tensor
        :return: Total quantile loss over the batch.
        :rtype: torch.Tensor
        """

        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.

    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        """
        Initializes the BertAdam optimizer with warmup and weight decay options.

        :param params: Model parameters to optimize.
        :type params: iterable
        :param lr: Learning rate.
        :type lr: float
        :param warmup: Proportion of warmup steps (-1 for no warmup).
        :type warmup: float
        :param t_total: Total training steps (-1 for constant lr).
        :type t_total: int
        :param schedule: Warmup schedule type (e.g., "warmup_linear").
        :type schedule: str
        :param b1: Exponential decay rate for first moment.
        :type b1: float
        :param b2: Exponential decay rate for second moment.
        :type b2: float
        :param e: Term added to denominator for numerical stability.
        :type e: float
        :param weight_decay: Weight decay coefficient.
        :type weight_decay: float
        :param max_grad_norm: Max gradient norm for clipping.
        :type max_grad_norm: float
        """

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self) -> List:
        """
        Computes the current learning rates for all parameter groups, applying
        a warmup schedule if specified.

        :return: A list of current learning rates, one per parameter group.
        :rtype: List[float]
        """

        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * \
                        schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """
        Performs a single optimization step using the BERT-adapted Adam optimizer,
        including gradient clipping, weight decay, and optional warmup scheduling.

        :param closure: Optional closure that reevaluates the model and returns the loss.
        :type closure: Callable, optional
        :return: The loss if closure is provided, otherwise None.
        :rtype: float or None
        """

        loss = None
        if closure is not None:
            loss = closure()

        warned_for_t_total = False

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step'] / group['t_total']
                    lr_scheduled = group['lr'] * schedule_fct(progress, group['warmup'])
                    # warning for exceeding t_total (only active with warmup_linear
                    if group['schedule'] == "warmup_linear" and progress > 1. and not warned_for_t_total:
                        logger.warning(
                            "Training beyond specified 't_total' steps with schedule '{}'. Learning rate set to {}. "
                            "Please set 't_total' of {} correctly.".format(
                                group['schedule'], lr_scheduled, self.__class__.__name__))
                        warned_for_t_total = True
                    # end warning
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


class NegativeLogLikelihood(torch.nn.Module):
    """target -> True y output -> predicted distribution."""

    def __init__(self):
        """
        Initializes NegativeLogLikelihood loss module.
        """

        super().__init__()

    def forward(self, output: torch.distributions, target: torch.Tensor):

        """
        Computes the negative log-likelihood from predicted distributions.

        :param output: Predicted probability distribution (e.g., Normal, Poisson).
        :type output: torch.distributions.Distribution
        :param target: Ground truth samples.
        :type target: torch.Tensor
        :return: Total negative log-likelihood loss.
        :rtype: torch.Tensor
        """

        return -output.log_prob(target).sum()


def l1_regularizer(model, lambda_l1=0.01):
    """
    Applies L1 regularization to the model weights.

    source: https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model

    :param model: The model to regularize.
    :type model: torch.nn.Module
    :param lambda_l1: Regularization strength.
    :type lambda_l1: float
    :return: L1 regularization loss.
    :rtype: torch.Tensor
    """

    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
        return lossl1


def orth_regularizer(model, lambda_orth=0.01):
    """
    Applies orthogonality regularization to the model weights.

    source: https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model

    :param model: The model to regularize.
    :type model: torch.nn.Module
    :param lambda_orth: Regularization strength.
    :type lambda_orth: float
    :return: Orthogonality regularization loss.
    :rtype: torch.Tensor
    """
    lossorth = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            param_flat = model_param_value.view(model_param_value.shape[0], -1)
            sym = torch.mm(param_flat, torch.t(param_flat))
            sym -= torch.eye(param_flat.shape[0])
            lossorth += lambda_orth * sym.sum()

        return lossorth


class InfoNCELoss(torch.nn.Module):
    """
    The InfoNCE contrastive loss for aligning paired embeddings (CLIP-style).

    Given a batch of anchor and positive embeddings where row i of each tensor describes the same
    entity (e.g. the visual and temporal embeddings of the same site), every other row in the batch
    serves as an in-batch negative. Embeddings are L2-normalized internally.
    """

    def __init__(self, temperature: float = 0.07, symmetric: bool = True):
        """
        Initializes the InfoNCE loss.

        :param temperature: The softmax temperature scaling the cosine similarities, defaults to 0.07.
        :type temperature: float
        :param symmetric: Whether to average the anchor->positive and positive->anchor directions,
            defaults to True.
        :type symmetric: bool
        """
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """
        Computes the contrastive loss over a batch of paired embeddings.

        :param anchor: Anchor embeddings of shape (batch_size, dim).
        :type anchor: torch.Tensor
        :param positive: Positive embeddings of shape (batch_size, dim), row-aligned with the anchors.
        :type positive: torch.Tensor
        :return: The scalar InfoNCE loss.
        :rtype: torch.Tensor
        """
        anchor = torch.nn.functional.normalize(anchor, dim=-1)
        positive = torch.nn.functional.normalize(positive, dim=-1)
        logits = anchor @ positive.t() / self.temperature
        targets = torch.arange(anchor.shape[0], device=anchor.device)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if self.symmetric:
            loss = (loss + torch.nn.functional.cross_entropy(logits.t(), targets)) / 2.0
        return loss


class NSELoss(torch.nn.Module):
    """
    Nash-Sutcliffe efficiency loss for streamflow: sum((sim - obs)^2) / sum((obs - mean(obs))^2).

    Equals 1 - NSE, so minimizing it maximizes the Nash-Sutcliffe efficiency. Values below 1 beat
    the mean-flow baseline. Computed per batch element and averaged.
    """

    def __init__(self, eps: float = 1e-6):
        """
        Initializes the NSE loss.

        :param eps: Stabilizer added to the observed variance, defaults to 1e-6.
        :type eps: float
        """
        super().__init__()
        self.eps = eps

    def forward(self, simulated: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """
        Computes 1 - NSE averaged over the batch.

        :param simulated: Simulated series of shape (batch_size, n_steps).
        :type simulated: torch.Tensor
        :param observed: Observed series of the same shape.
        :type observed: torch.Tensor
        :return: The scalar loss.
        :rtype: torch.Tensor
        """
        numerator = ((simulated - observed) ** 2).sum(dim=-1)
        denominator = ((observed - observed.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
        return (numerator / (denominator + self.eps)).mean()


class MaskedMSELoss(torch.nn.Module):
    """
    MSE over only the observed entries of a sparsely observed target (e.g. satellite ET passes).

    The mask is 1 where the target is observed and 0 elsewhere; gradients only flow on observed
    entries, implementing the masked-loss pattern for sparse supervision.
    """

    def forward(self, simulated: torch.Tensor, observed: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the masked MSE.

        :param simulated: Simulated values of any shape.
        :type simulated: torch.Tensor
        :param observed: Observed values of the same shape (entries where mask is 0 are ignored).
        :type observed: torch.Tensor
        :param mask: Binary observation mask of the same shape.
        :type mask: torch.Tensor
        :return: The scalar masked MSE (zero when nothing is observed).
        :rtype: torch.Tensor
        """
        squared = (simulated - observed) ** 2 * mask
        return squared.sum() / mask.sum().clamp(min=1.0)
