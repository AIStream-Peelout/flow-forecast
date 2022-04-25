import torch
from typing import List
from torch.autograd import Variable
from flood_forecast.model_dict_function import pytorch_criterion_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def numpy_to_tvar(x) -> torch.autograd.Variable:
    """ Converts a  numpy array into a PyTorch Tensor

    :param x: A numpy array you want to convert to a tensor
    :type x: torch.Tensor
    :return: A tensor variable
    :rtype: torch.Variable
    """
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))


def flatten_list_function(input_list: List) -> List:
    """
    A function to flatten a list.
    """
    return [item for sublist in input_list for item in sublist]


def make_criterion_functions(crit_list: List) -> List:
    """crit_list should be either dict or list
    returns a list
    """
    final_list = []
    if type(crit_list) == list:
        for crit in crit_list:
            final_list.append(pytorch_criterion_dict[crit]())
    else:
        for k, v in crit_list.items():
            final_list.append(pytorch_criterion_dict[k](**v))
    return final_list
