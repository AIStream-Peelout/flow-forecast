import torch
import math
from torch.nn.modules import Transformer, TransformerEncoder, TransformerEncoderLayer, LayerNorm
from torch.autograd import Variable


class SimpleTransformer(torch.nn.Module):
    def __init__(
            self,
            number_time_series: int,
            seq_length: int = 48,
            output_seq_len: int = None,
            d_model: int = 128,
            n_heads: int = 8,
            sigmoid=False):
        """
        Full transformer model
        """
        super().__init__()
        if output_seq_len is None:
            output_seq_len = seq_length
        self.out_seq_len = output_seq_len
        self.mask = generate_square_subsequent_mask(seq_length)
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.transformer = Transformer(d_model, nhead=n_heads)
        self.final_layer = torch.nn.Linear(d_model, 1)
        self.sequence_size = seq_length
        self.tgt_mask = generate_square_subsequent_mask(output_seq_len)
        self.sigmoid = None
        if sigmoid:
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, t: torch.Tensor, tgt_mask=None, src_mask=None):
        if src_mask:
            x = self.encode_sequence(x, src_mask)
        else:
            x = self.encode_sequence(x, src_mask)
        return self.decode_seq(x, t, tgt_mask)

    def basic_feature(self, x: torch.Tensor):
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        return x

    def encode_sequence(self, x, src_mask=None):
        x = self.basic_feature(x)
        x = self.transformer.encoder(x, src_mask)
        return x

    def decode_seq(self, mem, t, tgt_mask=None, view_number=None) -> torch.Tensor:
        if view_number is None:
            view_number = self.out_seq_len
        if tgt_mask is None:
            tgt_mask = self.tgt_mask
        t = self.basic_feature(t)
        x = self.transformer.decoder(t, mem, tgt_mask=tgt_mask)
        x = self.final_layer(x)
        if self.sigmoid:
            x = self.sigmoid(x)
        return x.view(-1, view_number)


class CustomTransformerDecoder(torch.nn.Module):
    def __init__(
            self,
            seq_length: int,
            output_seq_length: int,
            n_time_series: int,
            d_model=128,
            output_dim=1,
            n_layers_encoder=6,
            use_mask=False,
            n_heads=8):
        """
        Uses a number of encoder layers with simple linear decoder layer
        """
        super().__init__()
        self.dense_shape = torch.nn.Linear(n_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, 8)
        encoder_norm = LayerNorm(d_model)
        self.transformer_enc = TransformerEncoder(encoder_layer, n_layers_encoder, encoder_norm)
        self.output_dim_layer = torch.nn.Linear(d_model, output_dim)
        self.output_seq_length = output_seq_length
        self.out_length_lay = torch.nn.Linear(seq_length, output_seq_length)
        self.mask = generate_square_subsequent_mask(seq_length)
        self.mask_it = use_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass on tensor of (batch_size, sequence_length, n_time_series)
        Return tensor of dim (batch_size, output_seq_length)
        """
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)
        else:
            # Allow no mask
            x = self.transformer_enc(x)
        x = self.output_dim_layer(x)
        x = x.permute(1, 2, 0)
        x = self.out_length_lay(x)
        return x.view(-1, self.output_seq_length)


class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a basic positional encoding"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def greedy_decode(
        model,
        src: torch.Tensor,
        max_len: int,
        real_target: torch.Tensor,
        unsqueeze_dim=1,
        device='cpu'):
    """
    Mechanism to sequentially decode the model
    :src Historical time series values
    :real_target The real values (they should be masked), however if want can include known real values.
    :returns tensor
    """
    src = src.float()
    real_target = real_target.float()
    if hasattr(model, "mask"):
        src_mask = model.mask
    memory = model.encode_sequence(src, src_mask)
    # Get last element of src array to forecast from
    ys = src[:, -1, :].unsqueeze(unsqueeze_dim)
    for i in range(max_len):
        mask = generate_square_subsequent_mask(i + 1).to(device)
        with torch.no_grad():
            out = model.decode_seq(memory,
                                   Variable(ys),
                                   Variable(mask), i + 1)
            real_target[:, i, 0] = out[:, i]
            src = torch.cat((src, real_target[:, i, :].unsqueeze(1)), 1)
            ys = torch.cat((ys, real_target[:, i, :].unsqueeze(1)), 1)
        memory = model.encode_sequence(src[:, i + 1:, :], src_mask)
    return ys[:, 1:, :]
