def handle_csv_id_output(src, trg, model, criterion, random_sample=False):
    """A helper function to better handle the output of models with a series_id

    :param src: A dictionary of src sequences (partitioned by series_id)
    :type src: torch.Tensor
    :param trg: A dictionary of target sequences(partitioned by series_id)
    :type trg: torch.Tensor
    :param model: A model that takes both a src and a series_id
    :type model: [type]
    """
    total_loss = 0.0
    for (k, v), (k2, v2) in zip(src.items(), trg.items()):
        output = model(k, v)
        loss = criterion(output, v2)
        total_loss += loss.item()
    total_loss /= len(src.keys())
