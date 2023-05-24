from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from typing import Tuple, Dict
import numpy as np

attr_dict = {"IntegratedGradients": IntegratedGradients, "DeepLift": DeepLift, "GradientSHAP": GradientShap,
             "NoiseTunnel": NoiseTunnel, "FeatureAblation": FeatureAblation}


def run_attribution(model, test_loader, method, additional_params: Dict) -> Tuple:
    """Function that creates attribution for a model based on Captum.

    :param model: The deep learning model to be used for attribution. This should be a PyTorch model.
    :type model: _type_
    :param test_loader: Should be a FF CSVDataLoader or a related subclass.
    :type test_loader: _type_
    :param method: _description_
    :type method: _type_
    :return: Returns a Tuple of attributions and approximation error. This data is used to create plots.
    :rtype: Tuple
    """

    attribution_method = attr_dict[method](model)
    x, y = test_loader[0]
    attributions, approximation_error = attribution_method.attribute(x.unsqueeze(0), **additional_params)
    return attributions, approximation_error


def make_attribution_plots(x, attributions, approximation_error, feature_names, use_wandb: bool = True):
    """_summary_

    :param attributions: _description_
    :type attributions: _type_
    :param approximation_error: _description_
    :type approximation_error: _type_
    :param use_wandb: _description_, defaults to True
    :type use_wandb: bool, optional
    """
    x_axis_data = np.arange(x.shape[1])
    print(x_axis_data)
    print("Hello world")
