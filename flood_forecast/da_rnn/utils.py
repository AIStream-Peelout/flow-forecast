import logging
import os

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from flood_forecast.da_rnn.constants import device


def setup_log(tag='VOC_TOPICS'):
    """
    Sets up a basic logging configuration for the module.

    :param tag: The name to be assigned to the logger.
    :type tag: str
    :return: A configured Python logger instance.
    :rtype: logging.Logger
    """
    # create logger
    logger = logging.getLogger(tag)
    # logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)
    return logger


def save_or_show_plot(file_nm: str, save: bool, save_path=""):
    """
    Saves a matplotlib plot to a file or displays it based on the 'save' flag.

    :param file_nm: The filename to save the plot as, if saving.
    :type file_nm: str
    :param save: Boolean indicating whether to save the plot (True) or display it (False).
    :type save: bool
    :param save_path: The directory path where the plot should be saved. Defaults to current directory if empty.
    :type save_path: str
    :return: None
    :rtype: None
    """
    if save:
        plt.savefig(os.path.join(save_path, file_nm))
    else:
        plt.show()


def numpy_to_tvar(x: torch.Tensor):
    """
    Converts a numpy array (passed via torch.Tensor type hint) to a PyTorch FloatTensor Variable
    and moves it to the defined global device (CPU/GPU).

    :param x: The input data, expected to be a numpy array to be converted to a Tensor/Variable.
    :type x: torch.Tensor
    :return: A PyTorch Variable containing the input data as a FloatTensor on the appropriate device.
    :rtype: torch.autograd.Variable
    """
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))
