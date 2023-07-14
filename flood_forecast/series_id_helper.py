import torch


def handle_csv_id_output(src: torch.Tensor, trg: torch.Tensor, model, criterion, random_sample=False, n_targs=1):
    """A helper function to better handle the output of models with a series_id and compute loss,

    :param src: A dictionary of src sequences (partitioned by series_id)
    :type src: torch.Tensor
    :param trg: A dictionary of target sequences (partitioned by series_id)
    :type trg: torch.Tensor
    :param model: A model that takes both a src and a series_id
    :type model: torch.nn.Module
    """
    total_loss = 0.0
    for (k, v), (k2, v2) in zip(src.items(), trg.items()):
        output = model(v, k)
        loss = criterion(output, v2[:, :, :n_targs])
        total_loss += loss.item()
    total_loss /= len(src.keys())
    return total_loss
