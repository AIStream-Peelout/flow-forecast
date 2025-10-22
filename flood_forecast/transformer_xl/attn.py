import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import rearrange, repeat
import torch.nn.functional as F
from jaxtyping import Float
from torch import einsum
from typing import Tuple


class TriangularCausalMask:
    def __init__(self, B: int, L: int, device: str = "cpu"):
        """
        Creates a triangular causal mask tensor for self-attention mechanisms.

        :param B: Batch size.
        :type B: int
        :param L: Sequence length.
        :type L: int
        :param device: The device to place the mask tensor on (e.g., 'cpu' or 'cuda').
        :type device: str
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """
        Returns the triangular causal mask tensor.

        :return: The mask tensor of shape [B, 1, L, L].
        :rtype: torch.Tensor
        """
        return self._mask


class ProbMask:
    def __init__(
        self,
        B: int,
        H: int,
        L: int,
        index: torch.Tensor,
        scores: torch.Tensor,
        device: str = "cpu",
    ):
        """
        Creates a probabilistic mask used in ProbAttention (Informer).

        :param B: Batch size.
        :type B: int
        :param H: Number of attention heads.
        :type H: int
        :param L: Query sequence length (L_Q).
        :type L: int
        :param index: The indices of the top-k queries (M_top).
        :type index: torch.Tensor
        :param scores: The score tensor before softmax, used to determine the shape for the mask.
        :type scores: torch.Tensor
        :param device: The device to place the mask tensor on (e.g., 'cpu' or 'cuda').
        :type device: str
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """
        Returns the probabilistic mask tensor.

        :return: The mask tensor.
        :rtype: torch.Tensor
        """
        return self._mask


# Code implementation from https://github.com/thuml/Flowformer
class FlowAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1):
        """
        Implementation of Flow Attention from the Flowformer paper.

        :param attention_dropout: Dropout rate for attention weights.
        :type attention_dropout: float
        """
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a kernel function (sigmoid) to the input tensor.

        :param x: The input tensor (queries or keys).
        :type x: torch.Tensor
        :return: The tensor after applying the sigmoid kernel.
        :rtype: torch.Tensor
        """
        return torch.sigmoid(x)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        tau: float = None,
        delta: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the Flow Attention forward pass.

        :param queries: Query tensor of shape [B, L, H, D].
        :type queries: torch.Tensor
        :param keys: Key tensor of shape [B, S, H, D].
        :type keys: torch.Tensor
        :param values: Value tensor of shape [B, S, H, D].
        :type values: torch.Tensor
        :param attn_mask: Attention mask (not used in this implementation).
        :type attn_mask: torch.Tensor
        :param tau: Temperature parameter (not used in this implementation).
        :type tau: float
        :param delta: Delta parameter (not used in this implementation).
        :type delta: float
        :return: A tuple containing the context vector and None (as no attention is output).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # kernel
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6)
        )
        normalizer_col = 1.0 / (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6)
        )
        # reweighting
        normalizer_row_refine = torch.einsum(
            "nhld,nhd->nhl",
            queries + 1e-6,
            (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6,
        )
        normalizer_col_refine = torch.einsum(
            "nhsd,nhd->nhs",
            keys + 1e-6,
            (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6,
        )
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2]))
        )
        normalizer_col_refine = (
            torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]
        )  # B h L vis
        # multiply
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (
            (
                ((queries @ kv) * normalizer_row[:, :, :, None])
                * normalizer_row_refine[:, :, :, None]
            )
            .transpose(1, 2)
            .contiguous()
        )
        return x, None


# Code implementation from https://github.com/shreyansh26/FlashAttention-PyTorch
class FlashAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        """
        Implementation of Flash Attention using tiling and re-normalization (a simplified version).

        :param mask_flag: Whether to use an attention mask.
        :type mask_flag: bool
        :param factor: A factor parameter (not directly used in this implementation).
        :type factor: int
        :param scale: Scaling factor for the attention scores. If None, it's computed as 1/sqrt(D).
        :type scale: float
        :param attention_dropout: Dropout rate for attention weights.
        :type attention_dropout: float
        :param output_attention: Whether to output the attention matrix (not used in this implementation).
        :type output_attention: bool
        """
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def flash_attention_forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the block-wise Flash Attention forward pass.

        :param Q: Query tensor.
        :type Q: torch.Tensor
        :param K: Key tensor.
        :type K: torch.Tensor
        :param V: Value tensor.
        :type V: torch.Tensor
        :param mask: Attention mask tensor.
        :type mask: torch.Tensor
        :return: A tuple containing the output context vector (O), the normalization statistics (l3), and the maximum log-probability (m).
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        BLOCK_SIZE = 32
        NEG_INF = -1e10  # -infinity
        EPSILON = 1e-10
        # mask = torch.randint(0, 2, (128, 8)).to(device='cuda')
        O1 = torch.zeros_like(Q, requires_grad=True)
        l3 = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

        O1 = O1.to(device="cuda")
        l3 = l3.to(device="cuda")
        m = m.to(device="cuda")

        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)

        O_BLOCKS = list(torch.split(O1, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l3, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale

                S_ij = torch.einsum("... i d, ... j d -> ... i j", Qi_scaled, Kj)
                if mask is not None:
                    # Masking
                    maskj_temp = rearrange(maskj, "b j -> b 1 1 j")
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    # Masking
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.0)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

                P_ij_Vj = torch.einsum("... i j, ... j d -> ... i d", P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = (
                    torch.exp(mi - mi_new) * li
                    + torch.exp(m_block_ij - mi_new) * l_block_ij
                )

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                    torch.exp(m_block_ij - mi_new) / li_new
                ) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        l3 = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l3, m

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        tau: float = None,
        delta: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the Flash Attention forward pass.

        :param queries: Query tensor of shape [B, L, H, D].
        :type queries: torch.Tensor
        :param keys: Key tensor of shape [B, S, H, D].
        :type keys: torch.Tensor
        :param values: Value tensor of shape [B, S, H, D].
        :type values: torch.Tensor
        :param attn_mask: Attention mask tensor.
        :type attn_mask: torch.Tensor
        :param tau: Temperature parameter (not used in this implementation).
        :type tau: float
        :param delta: Delta parameter (not used in this implementation).
        :type delta: float
        :return: A tuple containing the context vector and None (as no attention is output).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        res = self.flash_attention_forward(
            queries.permute(0, 2, 1, 3),
            keys.permute(0, 2, 1, 3),
            values.permute(0, 2, 1, 3),  # noqa
            attn_mask,
        )[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float = None,
        attention_dropout: float = 0.1,
    ):
        """
        The full attention mechanism currently used by the Informer and ITransformer models.

        :param mask_flag: Whether to mask the attention mechanism.
        :type mask_flag: bool
        :param factor: The factor to use in the attention mechanism (not used in this implementation).
        :type factor: int
        :param scale: The scale to use in the attention mechanism. If None, it's computed as 1/sqrt(E).
        :type scale: float
        :param attention_dropout: The dropout to use in the attention mechanism.
        :type attention_dropout: float
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: TriangularCausalMask = None,
        tau: float = None,
        delta: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the Full Attention forward pass.

        :param queries: Query tensor of shape [B, L, H, E].
        :type queries: torch.Tensor
        :param keys: Key tensor of shape [B, S, H, E].
        :type keys: torch.Tensor
        :param values: Value tensor of shape [B, S, H, D].
        :type values: torch.Tensor
        :param attn_mask: Attention mask object, typically TriangularCausalMask for causal attention.
        :type attn_mask: TriangularCausalMask
        :param tau: Temperature parameter (not used in this implementation).
        :type tau: float
        :param delta: Delta parameter (not used in this implementation).
        :type delta: float
        :return: A tuple containing the context vector and the attention matrix.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A


# Code implementation from https://github.com/zhouhaoyi/Informer2020
class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        """
        Implementation of Probabilistic Sparse Self-Attention from the Informer paper.

        :param mask_flag: Whether to use a causal mask for the attention mechanism.
        :type mask_flag: bool
        :param factor: The constant 'c' used to determine the sparse attention sampling size: c*ln(L_k) and c*ln(L_q).
        :type factor: int
        :param scale: The scale to use in the attention mechanism. If None, it's computed as 1/sqrt(D).
        :type scale: float
        :param attention_dropout: The dropout to use in the attention mechanism.
        :type attention_dropout: float
        :param output_attention: Whether to output the full attention matrix (sparse + context).
        :type output_attention: bool
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(
        self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # n_top: c*ln(L_q)
        """
        Computes the sparse QK scores by sampling keys and selecting the top queries.

        :param Q: Query tensor of shape [B, H, L_Q, D].
        :type Q: torch.Tensor
        :param K: Key tensor of shape [B, H, L_K, E].
        :type K: torch.Tensor
        :param sample_k: Number of keys to sample for initial sparsity measurement (U_part).
        :type sample_k: int
        :param n_top: Number of top queries to select (u).
        :type n_top: int
        :return: A tuple containing the scores for the top-u queries and the indices of those top queries.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        """
        Initializes the context vector.

        :param V: Value tensor of shape [B, H, L_V, D].
        :type V: torch.Tensor
        :param L_Q: Query sequence length (L_Q).
        :type L_Q: int
        :return: The initial context tensor.
        :rtype: torch.Tensor
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert L_Q == L_V
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(
        self,
        context_in: torch.Tensor,
        V: torch.Tensor,
        scores: torch.Tensor,
        index: torch.Tensor,
        L_Q: int,
        attn_mask: ProbMask = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the context vector with attention over the selected top queries.

        :param context_in: The initial context tensor.
        :type context_in: torch.Tensor
        :param V: Value tensor of shape [B, H, L_V, D].
        :type V: torch.Tensor
        :param scores: QK scores for the selected top queries, shape [B, H, u, L_K].
        :type scores: torch.Tensor
        :param index: Indices of the top-u queries (M_top).
        :type index: torch.Tensor
        :param L_Q: Query sequence length (L_Q).
        :type L_Q: int
        :param attn_mask: Probabilistic mask object for causal attention.
        :type attn_mask: ProbMask
        :return: A tuple containing the updated context vector and the attention matrix (or None).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: ProbMask = None,
        tau: float = None,
        delta: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the Probabilistic Attention forward pass.

        :param queries: Query tensor of shape [B, L_Q, H, D].
        :type queries: torch.Tensor
        :param keys: Key tensor of shape [B, L_K, H, D].
        :type keys: torch.Tensor
        :param values: Value tensor of shape [B, L_K, H, D].
        :type values: torch.Tensor
        :param attn_mask: Attention mask object, typically ProbMask for causal attention.
        :type attn_mask: ProbMask
        :param tau: Temperature parameter (not used in this implementation).
        :type tau: float
        :param delta: Delta parameter (not used in this implementation).
        :type delta: float
        :return: A tuple containing the context vector and the attention matrix (or None).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None,
    ):
        """
        A wrapper layer that performs the projection of Q, K, V and then applies an attention mechanism.

        :param attention: The inner attention mechanism (e.g., FullAttention, ProbAttention).
        :type attention: nn.Module
        :param d_model: The input/output dimension of the model.
        :type d_model: int
        :param n_heads: The number of attention heads.
        :type n_heads: int
        :param d_keys: The dimension of keys and queries per head. Defaults to d_model // n_heads.
        :type d_keys: int
        :param d_values: The dimension of values per head. Defaults to d_model // n_heads.
        :type d_values: int
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        tau: float = None,
        delta: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the Attention Layer forward pass.

        :param queries: Query tensor of shape [B, L, D_model].
        :type queries: torch.Tensor
        :param keys: Key tensor of shape [B, S, D_model].
        :type keys: torch.Tensor
        :param values: Value tensor of shape [B, S, D_model].
        :type values: torch.Tensor
        :param attn_mask: Attention mask.
        :type attn_mask: torch.Tensor
        :param tau: Temperature parameter (passed to inner attention).
        :type tau: float
        :param delta: Delta parameter (passed to inner attention).
        :type delta: float
        :return: A tuple containing the context vector and the attention matrix (or None).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None,
        causal: bool = False,
        bucket_size: int = 4,
        n_hashes: int = 4,
    ):
        """
        A layer that wraps the LSHSelfAttention mechanism, handling sequence length padding for Reformer.

        :param attention: The attention mechanism (LSHSelfAttention is imported inside).
        :type attention: nn.Module
        :param d_model: The input/output dimension of the model.
        :type d_model: int
        :param n_heads: The number of attention heads.
        :type n_heads: int
        :param d_keys: Key dimension per head (ignored, handled by LSHSelfAttention).
        :type d_keys: int
        :param d_values: Value dimension per head (ignored, handled by LSHSelfAttention).
        :type d_values: int
        :param causal: Whether to use causal attention.
        :type causal: bool
        :param bucket_size: The size of the buckets for LSH.
        :type bucket_size: int
        :param n_hashes: The number of hash rounds for LSH.
        :type n_hashes: int
        """
        super().__init__()
        import LSHSelfAttention

        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
        )

    def fit_length(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Pads the queries tensor so its sequence length is a multiple of (bucket_size * 2).

        :param queries: The input tensor of shape [B, N, C].
        :type queries: torch.Tensor
        :return: The padded queries tensor.
        :rtype: torch.Tensor
        """
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat(
                [queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1
            )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        tau: float,
        delta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the Reformer attention forward pass. Assumes keys=values=queries in the LSHAttention context.

        :param queries: Query tensor of shape [B, N, C].
        :type queries: torch.Tensor
        :param keys: Key tensor (not used by LSHSelfAttention in this configuration).
        :type keys: torch.Tensor
        :param values: Value tensor (not used by LSHSelfAttention in this configuration).
        :type values: torch.Tensor
        :param attn_mask: Attention mask (not used).
        :type attn_mask: torch.Tensor
        :param tau: Temperature parameter (not used).
        :type tau: float
        :param delta: Delta parameter (not used).
        :type delta: float
        :return: A tuple containing the output tensor and None (as no attention is output).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the input tensor's dimension elements in groups of two, used for Rotary Positional Embeddings.

    :param x: The input tensor. The last dimension is assumed to be an even-dimension for rotation.
    :type x: torch.Tensor
    :return: The tensor with elements rotated in the last dimension.
    :rtype: torch.Tensor
    """
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        """
        A pre-normalization wrapper (LayerNorm followed by a function/module).

        :param dim: The dimension of the feature to normalize.
        :type dim: int
        :param fn: The function or module to apply after normalization.
        :type fn: nn.Module
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Performs the PreNorm forward pass.

        :param x: The input tensor to be normalized.
        :type x: torch.Tensor
        :return: The output of the wrapped function/module after normalization.
        :rtype: torch.Tensor
        """
        return self.fn(self.norm(x), **kwargs)


class CrossPreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        """
        A pre-normalization wrapper specifically for cross-attention, using two LayerNorms (for source and target).

        :param dim: The dimension of the feature to normalize.
        :type dim: int
        :param fn: The cross-attention function or module to apply after normalization.
        :type fn: nn.Module
        """
        super().__init__()
        self.norm_src = nn.LayerNorm(dim)
        self.norm_tgt = nn.LayerNorm(dim)
        self.fn = fn

    def forward(
        self,
        ctx: torch.Tensor,
        src_pos_emb: torch.Tensor,
        ts: torch.Tensor,
        tgt_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the CrossPreNorm forward pass.

        :param ctx: The source context tensor.
        :type ctx: torch.Tensor
        :param src_pos_emb: Positional embedding for the source.
        :type src_pos_emb: torch.Tensor
        :param ts: The target sequence tensor.
        :type ts: torch.Tensor
        :param tgt_pos_emb: Positional embedding for the target.
        :type tgt_pos_emb: torch.Tensor
        :return: The output of the wrapped cross-attention function.
        :rtype: torch.Tensor
        """
        return self.fn(self.norm_src(ctx), src_pos_emb, self.norm_tgt(ts), tgt_pos_emb)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the Gated Exponential Linear Unit (GELU) Linear Unit.

        :param x: The input tensor of shape [..., 2*D].
        :type x: torch.Tensor
        :return: The result of the GEGLU operation of shape [..., D].
        :rtype: torch.Tensor
        """
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_glu: bool = True,
    ):
        """
        A standard FeedForward network with an optional Gated Linear Unit (GEGLU or GLU) activation.

        :param dim: The input and output dimension of the feature.
        :type dim: int
        :param hidden_dim: The dimension of the hidden layer.
        :type hidden_dim: int
        :param dropout: Dropout rate to apply in the network.
        :type dropout: float
        :param use_glu: Whether to use the Gated Linear Unit (GEGLU) for activation.
        :type use_glu: bool
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the FeedForward network forward pass.

        :param x: The input tensor.
        :type x: torch.Tensor
        :return: The output tensor after the feed-forward operation.
        :rtype: torch.Tensor
        """
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rotary: bool = True,
    ):
        """
        The self-attention mechanism used in the CrossVIVIT model.

        :param dim: The input dimension of the sequence.
        :type dim: int
        :param heads: The number of attention heads.
        :type heads: int
        :param dim_head: The dimension of the heads.
        :type dim_head: int
        :param dropout: Dropout rate for attention weights and output projection.
        :type dropout: float
        :param use_rotary: Whether to use Rotary Positional Embeddings (RoPE).
        :type use_rotary: bool
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the Self Attention forward pass.

        :param x: Sequence of shape [B, N, D].
        :type x: torch.Tensor
        :param pos_emb: Positional embedding of sequence's tokens of shape [B, N, D].
        :type pos_emb: torch.Tensor
        :return: A tuple containing the output tensor and the attention matrix.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        q = self.to_q(x)

        qkv = (q, *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), qkv
        )

        if self.use_rotary:
            # Used to map dimensions from dimension
            sin, cos = map(
                lambda t: repeat(t, "b n d -> (b h) n d", h=self.heads), pos_emb
            )
            dim_rotary = sin.shape[-1]

            # handle the case where rotary dimension < head dimension

            (q, q_pass), (k, k_pass) = map(
                lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k)
            )
            q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
            q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))

        dots = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out), attn


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rotary: bool = True,
    ):
        """
        This is the CrossAttention module primarily used in the CrossVIVIT paper.

        :param dim: The input dimension of the sequence.
        :type dim: int
        :param heads: The number of heads for the attention mechanism.
        :type heads: int
        :param dim_head: The dimension of the heads.
        :type dim_head: int
        :param dropout: Dropout rate for attention weights and output projection.
        :type dropout: float
        :param use_rotary: Whether to use Rotary Positional Embeddings (RoPE).
        :type use_rotary: bool
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # Maps the input dimension to the inner dimension
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        src: Float[torch.Tensor, ""],
        src_pos_emb: torch.Tensor,
        tgt: torch.Tensor,
        tgt_pos_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the CrossAttention module.

        :param src: The source sequence (K, V) tensor.
        :type src: Float[torch.Tensor, ""]
        :param src_pos_emb: Positional embedding for the source.
        :type src_pos_emb: torch.Tensor
        :param tgt: The target sequence (Q) tensor.
        :type tgt: torch.Tensor
        :param tgt_pos_emb: Positional embedding for the target.
        :type tgt_pos_emb: torch.Tensor
        :return: A tuple containing the output tensor and the attention matrix.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        q = self.to_q(tgt)

        qkv = (q, *self.to_kv(src).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), qkv
        )

        if self.use_rotary:
            # apply 2-d rotary embeddings to queries and keys

            sin_src, cos_src = map(
                lambda t: repeat(t, "b n d -> (b h) n d", h=self.heads), src_pos_emb
            )
            sin_tgt, cos_tgt = map(
                lambda t: repeat(t, "b n d -> (b h) n d", h=self.heads), tgt_pos_emb
            )
            dim_rotary = sin_src.shape[-1]

            # handle the case where rotary dimension < head dimension

            (q, q_pass), (k, k_pass) = map(
                lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k)
            )
            q = (q * cos_tgt) + (rotate_every_two(q) * sin_tgt)
            k = (k * cos_src) + (rotate_every_two(k) * sin_src)
            q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))

        dots = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out), attn