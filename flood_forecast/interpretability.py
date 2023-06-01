from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from typing import Tuple, Dict

attr_dict = {"IntegratedGradients": IntegratedGradients, "DeepLift": DeepLift, "GradientSHAP": GradientShap,
             "NoiseTunnel": NoiseTunnel, "FeatureAblation": FeatureAblation}


def run_attribution(model, test_loader, method, additional_params: Dict) -> Tuple:
    """Function that creates attribution for a model based on Captum.
    :param model: The deep learning model to be used for attribution. This should be a PyTorch model.
    :type model: _type_
    :param test_loader: Should be a FF CSVDataLoader or a related sub-class.
    :type test_loader: _type_
    :param method: _description_
    :type method: _type_
    :return: d
    :rtype: Tuple

    .. code-block:: python

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


def make_attribution_plots(attributions, approximation_error, use_wandb: bool = True):
    """_summary_

    :param attributions: _description_
    :type attributions: _type_
    :param approximation_error: _description_
    :type approximation_error: _type_
    :param use_wandb: _description_, defaults to True
    :type use_wandb: bool, optional
    """
    pass
