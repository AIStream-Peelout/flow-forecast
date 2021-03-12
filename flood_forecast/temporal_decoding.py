
import torch
from typing import List


def decoding_function(src: torch.Tensor, trg: torch.Tensor, know_columns: List, unknown_columns: List,
                      max_seq_len: int, temp_trg: torch.Tensor, temp_src: torch.Tensor):
    """This is function aims to generalize decoding for our temporal data loader models. Sepcifically it uses.
    Args:

    """
    if temp_src is None:
        pass
    else:
        pass
