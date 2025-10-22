import torch
from typing import List
from torch.autograd import Variable
from flood_forecast.model_dict_function import pytorch_criterion_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def numpy_to_tvar(x: torch.Tensor) -> torch.autograd.Variable:
    """Converts a numpy array into a PyTorch Tensor.

    :param x: A numpy array you want to convert to a PyTorch tensor
    :type x: torch.Tensor
    :return: A tensor variable
    :rtype: torch.autograd.Variable
    """
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))


def flatten_list_function(input_list: List) -> List:
    """A function to flatten a list.

    :param input_list: A list of lists to be flattened.
    :type input_list: List
    :return: A flattened list.
    :rtype: List
    """
    return [item for sublist in input_list for item in sublist]


def make_criterion_functions(crit_list: List) -> List:
    """Creates a list of PyTorch criterion (loss) functions based on the input list or dictionary.

    If a list is provided, it contains the names of the criteria (e.g., ["MSELoss"]).
    If a dict is provided, keys are the names and values are the keyword arguments for the criterion (e.g., {"L1Loss": {"reduction": "mean"}}).

    :param crit_list: A list of criterion names (str) or a dictionary mapping criterion names to their keyword arguments.
    :type crit_list: List
    :return: A list of initialized PyTorch criterion functions.
    :rtype: List
    """
    final_list = []
    if type(crit_list) == list:
        for crit in crit_list:
            final_list.append(pytorch_criterion_dict[crit]())
    else:
        for k, v in crit_list.items():
            final_list.append(pytorch_criterion_dict[k](**v))
    return final_list