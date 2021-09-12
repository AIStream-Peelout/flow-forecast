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
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th
        (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
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
        This implements the MASE loss function (e.g. MAE_MODEL/MAE_NAIEVE)
        """
        super(MASELoss, self).__init__()
        self.method_dict = {"mean": lambda x, y: torch.mean(x, 1).unsqueeze(1).repeat(1, y[1], 1)}
        self.baseline_method = self.method_dict[baseline_method]

    def forward(self, target: torch.Tensor, output: torch.Tensor, train_data: torch.Tensor, m=1) -> torch.Tensor:
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
    '''
    Returns RMSE using:
    target -> True y
    output -> Prediction by model
    source: https://discuss.pytorch.org/t/rmse-loss-function/16540/3
    '''

    def __init__(self, variance_penalty=0.0):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.variance_penalty = variance_penalty

    def forward(self, output: torch.Tensor, target: torch.Tensor):
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
    '''
    Returns MAPE using:
    target -> True y
    output -> Predtion by model
    '''

    def __init__(self, variance_penalty=0.0):
        super().__init__()
        self.variance_penalty = variance_penalty

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if len(output) > 1:
            return torch.mean(torch.abs(torch.sub(target, output) / target)) + \
                self.variance_penalty * torch.std(torch.sub(target, output))
        else:
            return torch.mean(torch.abs(torch.sub(target, output) / target))


class PenalizedMSELoss(torch.nn.Module):
    '''
    Returns MSE using:
    target -> True y
    output -> Predtion by model
    source: https://discuss.pytorch.org/t/rmse-loss-function/16540/3
    '''

    def __init__(self, variance_penalty=0.0):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.variance_penalty = variance_penalty

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return self.mse(target, output) + \
            self.variance_penalty * torch.std(torch.sub(target, output))


# Add custom loss function
class GaussianLoss(torch.nn.Module):
    def __init__(self, mu=0, sigma=0):
        """Compute the negative log likelihood of Gaussian Distribution
        From https://arxiv.org/abs/1907.00235
        """
        super(GaussianLoss, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        loss = - tdist.Normal(self.mu, self.sigma).log_prob(x)
        return torch.sum(loss) / (loss.size(0) * loss.size(1))


class QuantileLoss(torch.nn.Module):
    """From https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629"""

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
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
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
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
    """
    target -> True y
    output -> predicted distribution
    """
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.distributions, target: torch.Tensor):
        """
        calculates NegativeLogLikelihood
        """
        return -output.log_prob(target).sum()


def l1_regularizer(model, lambda_l1=0.01):
    """
    source: https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model
    """
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
        return lossl1


def orth_regularizer(model, lambda_orth=0.01):
    """
    source: https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model
    """
    lossorth = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            param_flat = model_param_value.view(model_param_value.shape[0], -1)
            sym = torch.mm(param_flat, torch.t(param_flat))
            sym -= torch.eye(param_flat.shape[0])
            lossorth += lambda_orth * sym.sum()

        return lossorth
