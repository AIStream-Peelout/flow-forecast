import torch
from typing import Dict, List


def handle_csv_id_output(src: Dict[int, torch.Tensor], trg: Dict[int, torch.Tensor], model, criterion, opt,
                         random_sample: bool = False, n_targs: int = 1):
    """A helper function to better handle the output of models with a series_id and compute full loss.

    :param src: A dictionary of src sequences (partitioned by series_id)
    :type src: torch.Tensor
    :param trg: A dictionary of target sequences (key as series_id)
    :type trg: torch.Tensor
    :param model: A model that takes both a src and a series_id
    :type model: torch.nn.Module
    """
    total_loss = 0.00
    for (k, v), (k2, v2) in zip(src.items(), trg.items()):
        output = model.model(v, k)
        loss = criterion(output, v2[:, :, :n_targs])
        total_loss += loss.item()
        loss.backward()
        opt.step()
    total_loss /= len(src.keys())
    return total_loss


def handle_csv_id_validation(src: Dict[int, torch.Tensor], trg: Dict[int, torch.Tensor], model: torch.nn.Module,
                             criterion: List, random_sample: bool = False, n_targs: int = 1, max_seq_len: int = 100):
    """Function handles validation of models with a series_id. Returns a dictionary of losses for each criterion.

    :param src: A dictionary of source sequences with series_id as key and the torch.Tensor as the value.
    :type src: Dict[int, torch.Tensor]
    :param trg: A dictionary of target sequences with series_id as key and the torch.Tensor as the value.
    :type trg: Dict[int, torch.Tensor]
    :param model: _description_
    :type model: torch.nn.Module
    :param criterion: The criterion use for computing the validation loss.
    :type criterion: List
    :param random_sample: Whether to randomly sample from the source and target sequences, defaults to False
    :type random_sample: bool, optional
    :param n_targs: The number of forecasting targets, defaults to 1
    :type n_targs: int, optional
    :param max_seq_len: The max length, that can possibly be generated, defaults to 100
    :type max_seq_len: int, optional
    :return: Returns a dictionary of losses for each criterion for each series_id
    :rtype: Dict[str, float]
    """
    scaled_crit = dict.fromkeys(criterion, 0)
    losses = [0] * len(criterion)
    losses[0] = 0
    for (k, v), (k2, v2) in zip(src.items(), trg.items()):
        output = model(v, k)
        for critt in criterion:
            loss = critt(output, v2[:, :, :n_targs])
            scaled_crit[critt] += loss.item()
    return scaled_crit
