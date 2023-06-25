from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from typing import Tuple, Dict
import numpy as np

attr_dict = {"IntegratedGradients": IntegratedGradients, "DeepLift": DeepLift, "GradientSHAP": GradientShap,
             "NoiseTunnel": NoiseTunnel, "FeatureAblation": FeatureAblation}


def run_attribution(model, test_loader, method, additional_params: Dict) -> Tuple:
    """Function that creates attribution for a model based on Captum library.
    :param model: The deep learning model to be used for attribution. This should be a PyTorch model.
    :type model: torch.nn.Module
    :param test_loader: Should be a FF CSVDataLoader or a related sub-class.
    :type test_loader: _type_
    :param method: _description_
    :type method: _type_
    :return: d
    :rtype: Tuple

    .. code-block:: python

        from flood_forecast.interpretability import run_attribution
        model = VanillaGRU(3, 128, 2, 1, 0.2)

    ..
    """

    attribution_method = attr_dict[method](model)
    x, y = test_loader[0]
    the_data = attribution_method.attribute(x.unsqueeze(0), **additional_params)
    if isinstance(the_data, tuple):
        attributions, approximation_error = the_data
    else:
        attributions = the_data
        approximation_error = None
    return attributions, approximation_error


def make_attribution_plots(attributions, approximation_error, model, x, y, use_wandb: bool = True):
    """Creates the attribution plots and logs them to wandb if use_wandb is True.

    :param attributions: A tensor of the attributions should be of dimension (batch_size, , n_features).
    :type attributions: torch.Tensor
    :param approximation_error: _description_
    :type approximation_error: _type_
    :param use_wandb: _description_, defaults to True
    :type use_wandb: bool, optional
    """
    x_axis_data = np.arange(x.shape[1])
    x_axis_data_labels = model.params["fea"]

    ig_attr_test_sum = attributions.detach().numpy().sum(0)
    ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

    lin_weight = model.lin1.weight[0].detach().numpy()
    y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)
    pass
