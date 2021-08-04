"""
This code is based on huggingface,
https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py

MIT License

Copyright (c) 2018 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OFS CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
# Arxiv Link https://arxiv.org/pdf/1907.00235.pdf


import numpy as np
import torch
import torch.nn as nn
import math
# from torch.distributions.normal import Normal
import copy
from torch.nn.parameter import Parameter
from typing import Dict
from flood_forecast.transformer_xl.lower_upper_config import activation_dict


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)

ACT_FNS = {
    'relu': nn.ReLU(),
    'swish': swish,
    'gelu': gelu
}


class Attention(nn.Module):
    def __init__(self, n_head, n_embd, win_len, scale, q_len, sub_len, sparse=None, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Attention, self).__init__()

        if(sparse):
            print('Activate log sparse!')
            mask = self.log_mask(win_len, sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        self.register_buffer('mask_tri', mask)
        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while(index >= 0):
                if((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor, activation="Softmax"):
        activation = activation_dict[activation](dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):

        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn


class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_embd, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))
        self.e = e

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.e)
        return self.g * x + self.b


class MLP(nn.Module):
    def __init__(self, n_state, n_embd, acf='relu'):
        super(MLP, self).__init__()
        n_embd = n_embd
        self.c_fc = Conv1D(n_state, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_state)
        self.act = ACT_FNS[acf]
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        hidden1 = self.act(self.c_fc(x))
        hidden2 = self.c_proj(hidden1)
        return self.dropout(hidden2)


class Block(nn.Module):
    def __init__(self, n_head, win_len, n_embd, scale, q_len, sub_len, additional_params: Dict):
        super(Block, self).__init__()
        n_embd = n_embd
        self.attn = Attention(n_head, n_embd, win_len, scale, q_len, sub_len, **additional_params)
        self.ln_1 = LayerNorm(n_embd)
        self.mlp = MLP(4 * n_embd, n_embd)
        self.ln_2 = LayerNorm(n_embd)

    def forward(self, x):
        attn = self.attn(x)
        ln1 = self.ln_1(x + attn)
        mlp = self.mlp(ln1)
        hidden = self.ln_2(ln1 + mlp)
        return hidden


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, n_time_series, n_head, sub_len, num_layer, n_embd,
                 forecast_history: int, dropout: float, scale_att, q_len, additional_params: Dict, seq_num=None):
        super(TransformerModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = n_time_series
        self.n_head = n_head
        self.seq_num = None
        if seq_num:
            self.seq_num = seq_num
            self.id_embed = nn.Embedding(seq_num, n_embd)
            nn.init.normal_(self.id_embed.weight, std=0.02)
        self.n_embd = n_embd
        self.win_len = forecast_history
        # The following is the implementation of this paragraph
        """ For positional encoding in Transformer, we use learnable position embedding.
        For covariates, following [3], we use all or part of year, month, day-of-the-week,
        hour-of-the-day, minute-of-the-hour, age and time-series-ID according to the granularities of datasets.
        age is the distance to the first observation in that time series [3]. Each of them except time series
        ID has only one dimension and is normalized to have zero mean and unit variance (if applicable).
        """
        self.po_embed = nn.Embedding(forecast_history, n_embd)
        self.drop_em = nn.Dropout(dropout)
        block = Block(n_head, forecast_history, n_embd + n_time_series, scale=scale_att,
                      q_len=q_len, sub_len=sub_len, additional_params=additional_params)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layer)])
        nn.init.normal_(self.po_embed.weight, std=0.02)

    def forward(self, series_id: int, x: torch.Tensor):
        """Runs  forward pass of the DecoderTransformer model.

        :param series_id:   ID of the time series
        :type series_id: int
        :param x: [description]
        :type x: torch.Tensor
        :return: [description]
        :rtype: [type]
        """
        batch_size = x.size(0)
        length = x.size(1)  # (Batch_size, length, input_dim)
        embedding_sum = torch.zeros(batch_size, length, self.n_embd).to(self.device)
        if self.seq_num:
            embedding_sum = torch.zeros(batch_size, length)
            embedding_sum = embedding_sum.fill_(series_id).type(torch.LongTensor).to(self.device)
            embedding_sum = self.id_embed(embedding_sum)
        print("shape below")
        print(embedding_sum.shape)
        print(x.shape)
        print(series_id)
        position = torch.tensor(torch.arange(length), dtype=torch.long).to(self.device)
        po_embedding = self.po_embed(position)
        embedding_sum[:] = po_embedding
        x = torch.cat((x, embedding_sum), dim=2)
        for block in self.blocks:
            x = block(x)
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, n_time_series: int, n_head: int, num_layer: int,
                 n_embd: int, forecast_history: int, dropout: float, q_len: int, additional_params: Dict,
                 activation="Softmax", forecast_length: int = None, scale_att: bool = False, seq_num1=None,
                 sub_len=1, mu=None):
        """
        Args:
            n_time_series: Number of time series present in input
            n_head: Number of heads in the MultiHeadAttention mechanism
            seq_num: The number of targets to forecast
            sub_len: sub_len of the sparse attention
            num_layer: The number of transformer blocks in the model.
            n_embd: The dimention of Position embedding and time series ID embedding
            forecast_history: The number of historical steps fed into the time series model
            dropout: The dropout for the embedding of the model.
            additional_params: Additional parameters used to initalize the attention model. Can inc
        """
        super(DecoderTransformer, self).__init__()
        self.transformer = TransformerModel(n_time_series, n_head, sub_len, num_layer, n_embd, forecast_history,
                                            dropout, scale_att, q_len, additional_params, seq_num=seq_num1)
        self.softplus = nn.Softplus()
        self.mu = torch.nn.Linear(n_time_series + n_embd, 1, bias=True)
        self.sigma = torch.nn.Linear(n_time_series + n_embd, 1, bias=True)
        self._initialize_weights()
        self.mu_mode = mu
        self.forecast_len_layer = None
        if forecast_length:
            self.forecast_len_layer = nn.Linear(forecast_history, forecast_length)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, series_id: int = None):
        """
        Args:
            x: Tensor of dimension (batch_size, seq_len, number_of_time_series)
            series_id: Optional id of the series in the dataframe. Currently  not supported
        Returns:
            Case 1: tensor of dimension (batch_size, forecast_length)
            Case 2: GLoss sigma and mu: tuple of ((batch_size, forecast_history, 1), (batch_size, forecast_history, 1))
        """
        h = self.transformer(series_id, x)
        mu = self.mu(h)
        sigma = self.sigma(h)
        if self.mu_mode:
            sigma = self.softplus(sigma)
            return mu, sigma
        if self.forecast_len_layer:
            # Swap to (batch_size, 1, features) for linear layer
            sigma = sigma.permute(0, 2, 1)
            # Output (batch_size, forecast_len_)
            sigma = self.forecast_len_layer(sigma).permute(0, 2, 1)
        return sigma.squeeze(2)
