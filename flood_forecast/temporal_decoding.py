
import torch
from typing import List


def decoding_function(model, src: torch.Tensor, trg: torch.Tensor, know_columns: List, unknown_columns: List,
                      max_seq_len: int, forecast_len: int, temp_trg: torch.Tensor, temp_src: torch.Tensor):
    """A function that manages the decoding of more complex model. Assumes src is a tensor of dimension
    (batch_size, sequence_length, n_features). n_features should be formatted with the targets first (e.g.)
    cfs, precip followed by the

    :param model: [description]
    :type model: [type]
    :param src: [description]
    :type src: torch.Tensor
    :param trg: [description]
    :type trg: torch.Tensor
    :param know_columns: [description]
    :type know_columns: List
    :param unknown_columns: [description]
    :type unknown_columns: List
    :param max_seq_len: [description]
    :type max_seq_len: int
    :param forecast_len: [description]
    :type forecast_len: int
    :param temp_trg: [description]
    :type temp_trg: torch.Tensor
    :param temp_src: [description]
    :type temp_src: torch.Tensor
    """
    if temp_src is None:
        for i in range(0, max_seq_len):
            output = model(src, f)
    else:
        pass
