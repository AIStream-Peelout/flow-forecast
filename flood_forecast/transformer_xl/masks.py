import torch


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates a square mask for the sequence, typically used in a decoder to prevent
    attending to subsequent positions (causal masking).

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).

    :param sz: The size of the square mask (sequence length).
    :type sz: int
    :return: A square mask tensor of shape (sz, sz) with -inf and 0.0 values.
    :rtype: torch.Tensor
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TriangularCausalMask(object):
    def __init__(self, bath_size: int, seq_len: int, device: str = "cpu"):
        """This is a mask for the attention mechanism, specifically a triangular causal mask.
        It ensures that a position can only attend to all previous positions and itself.

        :param bath_size: The batch size.
        :type bath_size: int
        :param seq_len: The sequence length (number of time steps).
        :type seq_len: int
        :param device: The device for the tensor (e.g., "cpu", "cuda"), defaults to "cpu".
        :type device: str
        """
        mask_shape = [bath_size, 1, seq_len, seq_len]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """Returns the stored triangular causal mask tensor.

        :return: The triangular causal mask. Shape: (batch_size, 1, seq_len, seq_len)
        :rtype: torch.Tensor
        """
        return self._mask


class ProbMask(object):
    def __init__(self, B: int, H: int, L: int, index: torch.Tensor, scores: torch.Tensor, device: str = "cpu"):
        """Creates a probabilistic sparsity mask, typically used in models like Informer.
        It focuses attention on a subset of the most relevant query-key pairs.

        :param B: The batch size.
        :type B: int
        :param H: The number of attention heads.
        :type H: int
        :param L: The sequence length.
        :type L: int
        :param index: The indices of the selected top-k query attention scores. Shape: (B, H, L_Q)
        :type index: torch.Tensor
        :param scores: The full attention scores tensor before applying the final mask. Shape: (B, H, L_Q, L_K)
        :type scores: torch.Tensor
        :param device: The device for the tensor, defaults to "cpu".
        :type device: str
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """Returns the stored probabilistic mask tensor.

        :return: The probabilistic mask. Shape: (B, H, L_Q, L_K)
        :rtype: torch.Tensor
        """
        return self._mask