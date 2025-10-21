import collections
import typing

import numpy as np


class TrainConfig(typing.NamedTuple):
    """
    Configuration parameters for training a model.

    :param T: The sequence length for the time series data.
     :type T: int
     :param train_size: The number of samples to use for training.
     :type train_size: int
     :param batch_size: The number of samples per batch during training.
     :type batch_size: int
     :param loss_func: The callable function to compute the training loss.
     :type loss_func: typing.Callable
      :return: None
      :rtype: None
    """
    T: int
    train_size: int
    batch_size: int
    loss_func: typing.Callable


class TrainData(typing.NamedTuple):
    """
    Container for training features and targets data.

    :param feats: The input features array.
     :type feats: np.ndarray
     :param targs: The target values array.
     :type targs: np.ndarray
      :return: None
      :rtype: None
    """
    feats: np.ndarray
    targs: np.ndarray


DaRnnNet = collections.namedtuple("DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"])
