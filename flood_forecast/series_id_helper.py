import torch
from typing import Dict


def handle_csv_id_output(src: Dict[int, torch.Tensor], trg: Dict[int, torch.Tensor], model, criterion,
                         random_sample: bool = False, n_targs: int = 1) -> float:
    """A helper function to better handle the output of models with a series_id and compute loss,

    :param src: A dictionary of src sequences (partitioned by series_id)
    :type src: torch.Tensor
    :param trg: A dictionary of target sequences (key as series_id)
    :type trg: torch.Tensor
    :param model: A model that takes both a src and a series_id
    :type model: torch.nn.Module
    """
    total_loss = 0.00
    for (k, v), (k2, v2) in zip(src.items(), trg.items()):
        print("Shape of v below")
        print(v.shape)
        output = model.model(v, k)
        loss = criterion(output, v2[:, :, :n_targs])
        total_loss += loss.item()
    total_loss /= len(src.keys())
    loss.backward()
    loss.step()
    return total_loss
