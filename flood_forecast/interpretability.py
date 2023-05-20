from captum.attr import IntegratedGradients


def run_attribution(model, test_loader, method):
    """
    """
    ig = IntegratedGradients(model)
    x, y = test_loader[0]
    attributions, approximation_error = ig.attribute((input1, input2),
                                                 baselines=(baseline1, baseline2),
                                                 method='gausslegendre',
                                                 return_convergence_delta=True)
    pass
