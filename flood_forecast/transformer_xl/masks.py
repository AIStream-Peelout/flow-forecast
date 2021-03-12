import torch


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """ Generates a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TriangularCausalMask(object):
    def __init__(self, bath_size: int, seq_len: int, device="cpu"):
        """This is a mask for the attention mechanism

        :param bath_size: The batch_size should be passed on init
        :type bath_size: int
        :param seq_len: Number of historical time steps.
        :type seq_len: int
        :param device: The device typically will be cpu, cuda, or tpu, defaults to "cpu"
        :type device: str, optional
        """
        mask_shape = [bath_size, 1, seq_len, seq_len]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask(object):
    def __init__(self, B: int, H, L, index, scores, device="cpu"):
        """Creates a probablistic mask

        :param B: batch_size
        :type B: int
        :param H: Number of heads
        :type H: int
        :param L: Sequence length
        :type L: in
        :param index: [description]s
        :type index: [type]
        :param scores: [description]
        :type scores: [type]
        :param device: [description], defaults to "cpu"
        :type device: str, optional
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
