from captum.attr import IntegratedGradients
from typing import Tuple

attr_dict = {}

def run_attribution(model, test_loader, method) -> Tuple:
    """
    """
    ig = IntegratedGradients(model)
    x, y = test_loader[0]
    attributions, approximation_error = ig.attribute(x,
                                                 baselines=(baseline1),
                                                 method='gausslegendre',
                                                 return_convergence_delta=True)
    return attributions, approximation_error
