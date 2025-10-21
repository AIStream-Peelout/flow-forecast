"""Model from Keita Kurita.

Not useable
https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/transformer_xl_from_scratch.ipynb
"""
import torch
from torch import nn
from typing import Optional, Dict, List


class MultiHeadAttention(nn.Module):
    def __init__(self, d_input: int, d_inner: int, n_heads: int = 4,
                 dropout: float = 0.1, dropouta: float = 0.):
        """
        Initializes the Multi-Head Attention module with a relative positional encoding
        scheme (Transformer-XL style).

        :param d_input: The input dimension ($d_{model}$).
        :type d_input: int
        :param d_inner: The dimension of the inner attention projection ($d_k$ or $d_v$).
        :type d_inner: int
        :param n_heads: The number of attention heads. Defaults to 4.
        :type n_heads: int
        :param dropout: Dropout probability for the output projection layer. Defaults to 0.1.
        :type dropout: float
        :param dropouta: Dropout probability for the attention scores. Defaults to 0.0.
        :type dropouta: float
        """
        super().__init__()
        self.d_input = d_input
        self.d_inner = d_inner
        self.n_heads = n_heads
        # this layer applies the linear transformation required
        # for the keys and values for all heads at once for efficiency
        self.linear_kv = nn.Linear(
            d_input,
            (d_inner * n_heads * 2),  # 2 is for keys and values
            bias=False,  # we don't apply bias, making this a simple matrix multiplication
        )
        # for queries (will not be concatenated with memorized states so separate)
        self.linear_q = nn.Linear(
            d_input, d_inner * n_heads,
            bias=False
        )
        # for positional embeddings
        self.linear_p = nn.Linear(
            d_input, d_inner * n_heads,
            bias=False
        )
        self.scale = 1 / (d_inner ** 0.5)  # for scaled dot product attention
        self.dropa = nn.Dropout(dropouta)
        # we will use this to project back to the input dimension
        self.lout = nn.Linear(self.d_inner * self.n_heads, self.d_input, bias=False)
        self.norm = nn.LayerNorm(self.d_input)
        self.dropo = nn.Dropout(dropout)

    def _rel_shift(self, x):
        """
        Performs the relative positional shifting operation as described in the
        Transformer-XL paper. This shifts the $A_{R,i,j}$ term (position-based attention)
        to align the relative positions correctly.

        :param x: The tensor to be shifted, typically of shape $(L_q, L_k, B, H)$.
        :type x: torch.FloatTensor
        :return: The relatively shifted tensor.
        :rtype: torch.FloatTensor
        """
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        return (torch.cat([zero_pad, x], dim=1)
                .view(x.size(1) + 1, x.size(0), *x.size()[2:])[1:]
                .view_as(x))

    def forward(self, input_: torch.FloatTensor,  # (cur_seq, b, d_in)
                pos_embs: torch.FloatTensor,  # (cur_seq + prev_seq, d_in)
                memory: torch.FloatTensor,  # (prev_seq, b, d_in)
                u: torch.FloatTensor,  # (H, d)
                v: torch.FloatTensor,  # (H, d)
                mask: Optional[torch.FloatTensor] = None,
                ):
        """
        Performs the forward pass of the Multi-Head Attention with relative positional encoding.

        :param input_: The current segment's input tensor of shape $(L_{cur}, B, d_{input})$.
        :type input_: torch.FloatTensor
        :param pos_embs: The absolute positional embeddings for the total sequence,
                         shape $(L_{cur} + L_{prev}, d_{input})$.
        :type pos_embs: torch.FloatTensor
        :param memory: The recurrent memory from the previous segment, shape $(L_{prev}, B, d_{input})$.
        :type memory: torch.FloatTensor
        :param u: The global content bias vector, shape $(H, d_{inner})$.
        :type u: torch.FloatTensor
        :param v: The global positional bias vector, shape $(H, d_{inner})$.
        :type v: torch.FloatTensor
        :param mask: An optional attention mask to apply to the attention scores,
                     shape typically $(L_{cur}, L_{cur} + L_{prev}, 1)$. Defaults to None.
        :type mask: Optional[torch.FloatTensor]
        :return: The output tensor after attention, residual connection, and layer normalization,
                 shape $(L_{cur}, B, d_{input})$.
        :rtype: torch.FloatTensor
        """
        cur_seq = input_.shape[0]  # sequence length of current segment
        prev_seq = memory.shape[0]  # sequence length of previous segment
        H, d = self.n_heads, self.d_inner
        input_with_memory = torch.cat([memory, input_], dim=0)  # concatenate recurrent memory
        # across sequence dimension

        # we will use the following symbols to represent the shape of the tensors
        # cs: current sequence length, b: batch, H: number of heads
        # d: inner dimension, ps: previous sequence length
        # The key and value are now conditioned on the preceding context
        k_tfmd, v_tfmd = \
            torch.chunk(self.linear_kv(input_with_memory), 2, dim=-1)  # (cs + ps, b, H * d)
        q_tfmd = self.linear_q(input_)  # (cs, b, H * d)

        # apply scaled dot product attention
        # look at the following dimensions carefully, since this is the key operation
        # in the Transformer/Transformer XL architecture

        _, bs, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[1]
        # content-based attention term ((a) + (c) in the paper)
        # this is the standard attention term in the original Transformer, except without positional embeddings
        # which are handled separately in the Transformer XL (see below)
        # here, i corresponds to the number of queries = number of current inputs/targets (seq-wise)
        # j corresponds to the number of key/values = number of vectors that we can use to compute the
        # vector for each query
        content_attn = torch.einsum("ibhd,jbhd->ijbh", (
            (q_tfmd.view(cur_seq, bs, H, d) +  # (a)
             u),  # (c): u represents the global (independent of the query)
            # bias towards certain key/values = words
            # Note: maybe this could be a per-attention head parameter?
            k_tfmd.view(cur_seq + prev_seq, bs, H, d)  # There is no positional information to be found here
        ))  # (cs, cs + ps, b, H)

        # position-based attention term ((b) + (d) in the paper)
        # this attention is solely based on the position of the key/values
        # (i.e. it does not take the content of the key/values into account)
        p_tfmd = self.linear_p(pos_embs)  # (cs + ps, b, H * d)
        position_attn = torch.einsum("ibhd,jhd->ijbh", (
            (q_tfmd.view(cur_seq, bs, H, d) +  # (b)
             v),  # (d): v represents the global (independent of the query)
            # bias towards certain positions
            p_tfmd.view(cur_seq + prev_seq, H, d)  # Notice there is not content information
            # regarding keys and values here!
        ))  # (cs, cs + ps, b, H)

        #  Compute positional attention efficiently
        position_attn = self._rel_shift(position_attn)

        # the attention is the sum of content-based and position-based attention
        attn = content_attn + position_attn

        if mask is not None and mask.any().item():
            attn = attn.masked_fill(
                mask[..., None], -float('inf'))
        attn = torch.softmax(attn * self.scale,  # rescale to prevent values from exploding
                             dim=1)  # normalize across the value sequence dimension
        attn = self.dropa(attn)

        attn_weighted_values = (torch.einsum("ijbh,jbhd->ibhd",
                                             (attn,  # (cs, cs + ps, b, H)
                                              v_tfmd.view(cur_seq + prev_seq, bs, H, d),  # (cs + ps, b, H, d)
                                              ))  # (cs, b, H, d)
                                .contiguous()  # we need to change the memory layout to make `view` work
                                .view(cur_seq, bs, H * d))  # (cs, b, H * d)

        # Project back to input dimension and add residual connection
        output = input_ + self.dropo(self.lout(attn_weighted_values))
        output = self.norm(output)
        return output


class PositionwiseFF(nn.Module):
    def __init__(self, d_input, d_inner, dropout):
        """
        Initializes the Position-wise Feed-Forward Network module.

        :param d_input: The input/output dimension ($d_{model}$).
        :type d_input: int
        :param d_inner: The inner dimension of the feed-forward network.
        :type d_inner: int
        :param dropout: Dropout probability for the feed-forward network layers.
        :type dropout: float
        """
        super().__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = nn.Sequential(
            nn.Linear(d_input, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_input),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, input_: torch.FloatTensor,  # (cur_seq, bs, d_input)
                ) -> torch.FloatTensor:  # (cur_seq, bs, d_input)
        """
        Performs the forward pass of the Position-wise Feed-Forward Network.

        :param input_: The input tensor, shape $(L_{cur}, B, d_{input})$.
        :type input_: torch.FloatTensor
        :return: The output tensor after the feed-forward, residual connection, and layer normalization,
                 shape $(L_{cur}, B, d_{input})$.
        :rtype: torch.FloatTensor
        """
        ff_out = self.ff(input_)
        output = self.layer_norm(input_ + ff_out)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, d_input,
                 d_head_inner, d_ff_inner,
                 dropout, dropouta=0.):
        """
        Initializes a single Decoder Block, consisting of Multi-Head Attention and Position-wise FF.

        :param n_heads: The number of attention heads.
        :type n_heads: int
        :param d_input: The input dimension ($d_{model}$).
        :type d_input: int
        :param d_head_inner: The inner dimension of the attention projection.
        :type d_head_inner: int
        :param d_ff_inner: The inner dimension of the feed-forward network.
        :type d_ff_inner: int
        :param dropout: Dropout probability for the output projection in MHA and layers in FF.
        :type dropout: float
        :param dropouta: Dropout probability for the attention scores in MHA. Defaults to 0.0.
        :type dropouta: float
        """
        super().__init__()
        self.mha = MultiHeadAttention(d_input, d_head_inner, n_heads=n_heads,
                                      dropout=dropout, dropouta=dropouta)
        self.ff = PositionwiseFF(d_input, d_ff_inner, dropout)

    def forward(self, input_: torch.FloatTensor,  # (cur_seq, bs, d_input)
                pos_embs: torch.FloatTensor,  # (cur_seq + prev_seq, d_input),
                u: torch.FloatTensor,  # (H, d_input),
                v: torch.FloatTensor,  # (H, d_input),
                mask=None,
                mems=None,
                ):
        """
        Performs the forward pass of the Decoder Block.

        :param input_: The current segment's input tensor of shape $(L_{cur}, B, d_{input})$.
        :type input_: torch.FloatTensor
        :param pos_embs: The absolute positional embeddings, shape $(L_{cur} + L_{prev}, d_{input})$.
        :type pos_embs: torch.FloatTensor
        :param u: The global content bias vector, shape $(H, d_{inner})$.
        :type u: torch.FloatTensor
        :param v: The global positional bias vector, shape $(H, d_{inner})$.
        :type v: torch.FloatTensor
        :param mask: An optional attention mask. Defaults to None.
        :type mask: Optional[torch.FloatTensor]
        :param mems: The recurrent memory from the previous segment, shape $(L_{prev}, B, d_{input})$. Defaults to None.
        :type mems: Optional[torch.FloatTensor]
        :return: The output tensor after the attention and feed-forward layers.
        :rtype: torch.FloatTensor
        """
        return self.ff(self.mha(input_, pos_embs, mems, u, v, mask=mask))


class PositionalEmbedding(nn.Module):
    def __init__(self, d):
        """
        Initializes the Sinusoidal Positional Embedding module.

        :param d: The embedding dimension ($d_{model}$).
        :type d: int
        """
        super().__init__()
        self.d = d
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d, 2.0) / d))
        # register buffer tells pytorch that this tensor is part of the modle
        # this means that it will be saved in the state_dict and moved to the GPU
        # along with the model
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions: torch.LongTensor,  # (seq, )
                ):
        """
        Calculates the sinusoidal positional embeddings for a sequence of positions.

        :param positions: A tensor of integer positions, shape $(L, )$.
        :type positions: torch.LongTensor
        :return: The positional embedding tensor, shape $(L, 1, d_{model})$.
        :rtype: torch.FloatTensor
        """
        # outer product
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class StandardWordEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, div_val=1, sample_softmax=False):
        """
        Initializes the standard scaled Word Embedding layer.

        :param num_embeddings: The size of the vocabulary.
        :type num_embeddings: int
        :param embedding_dim: The dimension of the embeddings ($d_{model}$).
        :type embedding_dim: int
        :param div_val: Parameter for adaptive softmax (not used in this implementation). Defaults to 1.
        :type div_val: int
        :param sample_softmax: Flag for sampled softmax (not used in this implementation). Defaults to False.
        :type sample_softmax: bool
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.scale = embedding_dim ** 0.5

    def forward(self, input_: torch.LongTensor):
        """
        Performs the forward pass to retrieve and scale word embeddings.

        :param input_: The input token IDs, shape $(L, B)$.
        :type input_: torch.LongTensor
        :return: The scaled word embeddings, shape $(L, B, d_{model})$.
        :rtype: torch.FloatTensor
        """
        return self.embedding(input_) * self.scale


class TransformerXL(torch.nn.Module):
    def __init__(self, num_embeddings, n_layers, n_heads,
                 d_model, d_head_inner, d_ff_inner,
                 dropout=0.1, dropouta=0.,
                 seq_len: int = 0, mem_len: int = 0):
        """
        Initializes the Transformer-XL model.

        :param num_embeddings: The size of the vocabulary.
        :type num_embeddings: int
        :param n_layers: The number of decoder blocks.
        :type n_layers: int
        :param n_heads: The number of attention heads.
        :type n_heads: int
        :param d_model: The main model dimension ($d_{model}$).
        :type d_model: int
        :param d_head_inner: The inner dimension of the attention projection.
        :type d_head_inner: int
        :param d_ff_inner: The inner dimension of the feed-forward network.
        :type d_ff_inner: int
        :param dropout: Dropout probability for layers. Defaults to 0.1.
        :type dropout: float
        :param dropouta: Dropout probability for attention scores. Defaults to 0.0.
        :type dropouta: float
        :param seq_len: The segment sequence length. Defaults to 0.
        :type seq_len: int
        :param mem_len: The memory length. Defaults to 0.
        :type mem_len: int
        """
        super().__init__()
        self.n_layers, self.n_heads, self.d_model, self.d_head_inner, self.d_ff_inner = \
            n_layers, n_heads, d_model, d_head_inner, d_ff_inner
        # Embedding layers
        self.word_embs = StandardWordEmbedding(num_embeddings, d_model)
        self.pos_embs = PositionalEmbedding(d_model)
        # Core transformer
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderBlock(n_heads, d_model, d_head_inner=d_head_inner,
                                                  d_ff_inner=d_ff_inner,
                                                  dropout=dropout, dropouta=dropouta)
                                     for _ in range(n_layers)])

        # tie weights
        self.output_projection = nn.Linear(d_model, num_embeddings)
        self.output_projection.weight = self.word_embs.embedding.weight
        self.loss_fn = nn.CrossEntropyLoss()

        self.seq_len, self.mem_len = seq_len, mem_len

        # u and v are global parameters: maybe changing these to per-head parameters
        # might help performance?
        self.u, self.v = (nn.Parameter(torch.Tensor(self.n_heads, self.d_head_inner)),
                          nn.Parameter(torch.Tensor(self.n_heads, self.d_head_inner)))

    def init_memory(self, device=torch.device("cpu")) -> List[torch.FloatTensor]:
        """
        Initializes the recurrent memory with empty tensors for each layer plus the input.

        :param device: The device to create the tensors on. Defaults to torch.device("cpu").
        :type device: torch.device
        :return: A list of empty memory tensors.
        :rtype: List[torch.FloatTensor]
        """
        return [torch.empty(0, dtype=torch.float).to(device) for _ in range(self.n_layers + 1)]

    def update_memory(self,
                      previous_memory: List[torch.FloatTensor],
                      hidden_states: List[torch.FloatTensor],
                      ) -> List[torch.FloatTensor]:
        """
        Updates the recurrent memory by concatenating the previous memory with the current
        hidden states and keeping only the most recent `self.mem_len` steps.

        :param previous_memory: A list of memory tensors from the previous segment,
                                shape $(L_{prev}, B, d_{model})$.
        :type previous_memory: List[torch.FloatTensor]
        :param hidden_states: A list of the hidden states (inputs and outputs of each block)
                              from the current segment, shape $(L_{cur}, B, d_{model})$.
        :type hidden_states: List[torch.FloatTensor]
        :return: A list of updated memory tensors, shape $(L_{mem}, B, d_{model})$.
        :rtype: List[torch.FloatTensor]
        """
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)

        # For the updated memory, we use the most recent `self.mem_len`
        # states, including the previous memory
        # In other words, if `seq_len` < `self.mem_len` some of the previous memory
        # will carry over to the next memory o
        with torch.no_grad():
            new_memory = []
            end_idx = mem_len + seq_len
            beg_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(previous_memory, hidden_states):
                cat = torch.cat([m, h], dim=0)  # (mem_len + seq_len, bs, d)
                new_memory.append(cat[beg_idx:end_idx].detach())  # (self.mem_len, bs, d)
        return new_memory

    def reset_length(self, seq_len, ext_len, mem_len):
        """
        Resets the sequence length and memory length parameters.

        :param seq_len: The new segment sequence length.
        :type seq_len: int
        :param ext_len: The new extension length (not used in this implementation).
        :type ext_len: int
        :param mem_len: The new memory length.
        :type mem_len: int
        """
        self.seq_len = seq_len
        self.mem_len = mem_len

    def forward(self, idxs: torch.LongTensor,  # (cs, bs) 2
                target: torch.LongTensor,  # (cs, bs)
                memory: Optional[List[torch.FloatTensor]] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass of the Transformer-XL model, including memory update.

        :param idxs: The input token IDs for the current segment, shape $(L_{cur}, B)$.
        :type idxs: torch.LongTensor
        :param target: The target token IDs for loss calculation, shape $(L_{cur}, B)$.
        :type target: torch.LongTensor
        :param memory: The recurrent memory from the previous segment, list of tensors
                       of shape $(L_{prev}, B, d_{model})$. Defaults to None.
        :type memory: Optional[List[torch.FloatTensor]]
        :return: A dictionary containing the computed loss, logits, and the new memory.
        :rtype: Dict[str, torch.Tensor]
        """
        if memory is None:
            memory: List[torch.FloatTensor] = self.init_memory(idxs.device)
        assert len(memory) == len(self.layers) + 1
        cur_seq, bs = idxs.size()
        prev_seq = memory[0].size(0)

        # Construct the attention mask
        dec_attn_mask = torch.triu(
            torch.ones((cur_seq, cur_seq + prev_seq)),
            diagonal=1 + prev_seq,
        ).byte()[..., None].to(idxs.device)

        word_embs = self.drop(self.word_embs(idxs))
        pos_idxs = torch.arange(cur_seq + prev_seq - 1, -1, -1.0,
                                dtype=torch.float).to(word_embs.device)
        pos_embs = self.drop(self.pos_embs(pos_idxs))

        # Main part of forward passe
        hidden_states = [word_embs]
        layer_out = word_embs
        for mem, layer in zip(memory, self.layers):
            layer_out = layer(layer_out, pos_embs, self.u, self.v,
                              mask=dec_attn_mask, mems=mem)
            hidden_states.append(layer_out)

        logits = self.output_projection(self.drop(layer_out))
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))

        # Update the memory
        # Ensure the memory is treated as a constant
        # and we do not back propagate through them
        new_memory = self.update_memory(memory, hidden_states)
        return {"loss": loss, "logits": logits, "memory": new_memory}