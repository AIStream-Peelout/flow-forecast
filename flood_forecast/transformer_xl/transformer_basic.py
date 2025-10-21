import torch
import math
from torch.nn.modules import Transformer, TransformerEncoder, TransformerEncoderLayer, LayerNorm
from flood_forecast.transformer_xl.masks import generate_square_subsequent_mask
from torch.autograd import Variable
from flood_forecast.meta_models.merging_model import MergingModel
from flood_forecast.transformer_xl.lower_upper_config import activation_dict


class SimpleTransformer(torch.nn.Module):
    def __init__(
            self,
            number_time_series: int,
            seq_length: int = 48,
            output_seq_len: int = None,
            d_model: int = 128,
            n_heads: int = 8,
            dropout=0.1,
            forward_dim=2048,
            sigmoid=False):
        """A full transformer model.

        :param number_time_series: The total number of time series present
            (e.g. n_feature_time_series + n_targets)
        :type number_time_series: int
        :param seq_length: The length of your input sequence, defaults to 48
        :type seq_length: int
        :param output_seq_len: The length of your output sequence, defaults
            to None
        :type output_seq_len: int or None
        :param d_model: The dimensions of your model, defaults to 128
        :type d_model: int
        :param n_heads: The number of heads in each encoder/decoder block,
            defaults to 8
        :type n_heads: int
        :param dropout: The fraction of dropout you wish to apply during
            training, defaults to 0.1 (currently not functional)
        :type dropout: float
        :param forward_dim: Currently not functional, defaults to 2048
        :type forward_dim: int
        :param sigmoid: Whether to apply a sigmoid activation to the final
            layer (useful for binary classification), defaults to False
        :type sigmoid: bool
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
        """
        Performs the forward pass for the Transformer model.

        :param x: The source sequence tensor (input to the encoder).
        :type x: torch.Tensor
        :param t: The target sequence tensor (input to the decoder).
        :type t: torch.Tensor
        :param tgt_mask: An optional mask for the target sequence in the decoder.
        :type tgt_mask: torch.Tensor or None
        :param src_mask: An optional mask for the source sequence in the encoder.
        :type src_mask: torch.Tensor or None
        :return: The output tensor from the decoder after the final linear layer and optional sigmoid.
        :rtype: torch.Tensor
        """
        x = self.encode_sequence(x[:, :-1, :], src_mask)
        return self.decode_seq(x, t, tgt_mask)

    def basic_feature(self, x: torch.Tensor):
        """
        Applies the initial linear layer and positional encoding.

        :param x: The input tensor.
        :type x: torch.Tensor
        :return: The transformed and permuted tensor with positional encoding.
        :rtype: torch.Tensor
        """
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        return x

    def encode_sequence(self, x, src_mask=None):
        """
        Encodes the input sequence using the Transformer encoder.

        :param x: The source sequence tensor.
        :type x: torch.Tensor
        :param src_mask: An optional mask for the source sequence.
        :type src_mask: torch.Tensor or None
        :return: The output tensor from the Transformer encoder (memory).
        :rtype: torch.Tensor
        """
        x = self.basic_feature(x)
        x = self.transformer.encoder(x, src_mask)
        return x

    def decode_seq(self, mem, t, tgt_mask=None, view_number=None) -> torch.Tensor:
        """
        Decodes the sequence using the Transformer decoder.

        :param mem: The memory tensor from the encoder output.
        :type mem: torch.Tensor
        :param t: The target sequence tensor.
        :type t: torch.Tensor
        :param tgt_mask: An optional mask for the target sequence in the decoder.
        :type tgt_mask: torch.Tensor or None
        :param view_number: The length to flatten the output sequence to. Defaults to `self.out_seq_len`.
        :type view_number: int or None
        :return: The final output tensor of shape (-1, view_number).
        :rtype: torch.Tensor
        """
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
            forward_dim=2048,
            dropout=0.1,
            use_mask=False,
            meta_data=None,
            final_act=None,
            squashed_embedding=False,
            n_heads=8):
        """Uses a number of encoder layers with simple linear decoder layer.

        :param seq_length: The number of historical time-steps fed into the model in each forward pass.
        :type seq_length: int
        :param output_seq_length: The number of forecasted time-steps outputted by the model.
        :type output_seq_length: int
        :param n_time_series: The total number of time series present (targets + features)
        :type n_time_series: int
        :param d_model: The embedding dim of the mode, defaults to 128
        :type d_model: int
        :param output_dim: The output dimension (should correspond to n_targets), defaults to 1
        :type output_dim: int
        :param n_layers_encoder: The number of encoder layers, defaults to 6
        :type n_layers_encoder: int
        :param forward_dim: The forward embedding dim, defaults to 2048
        :type forward_dim: int
        :param dropout: How much dropout to use, defaults to 0.1
        :type dropout: float
        :param use_mask: Whether to use subsquent sequence mask during training, defaults to False
        :type use_mask: bool
        :param meta_data: Configuration for static meta-data merging, defaults to None. Expected to be a dict if used.
        :type meta_data: dict or None
        :param final_act: The name of the final activation function (e.g., 'sigmoid'), looked up in `activation_dict`, defaults to None.
        :type final_act: str or None
        :param squashed_embedding: Whether to create a one 1-D time embedding by squashing the sequence dimension, defaults to False.
        :type squashed_embedding: bool
        :param n_heads: The number of attention heads in the encoder layer, defaults to 8
        :type n_heads: int
        """
        super().__init__()
        self.dense_shape = torch.nn.Linear(n_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, 8, forward_dim, dropout)
        encoder_norm = LayerNorm(d_model)
        self.transformer_enc = TransformerEncoder(encoder_layer, n_layers_encoder, encoder_norm)
        self.output_dim_layer = torch.nn.Linear(d_model, output_dim)
        self.output_seq_length = output_seq_length
        self.out_length_lay = torch.nn.Linear(seq_length, output_seq_length)
        self.mask = generate_square_subsequent_mask(seq_length)
        self.out_dim = output_dim
        self.mask_it = use_mask
        self.final_act = None
        self.squashed = None
        if final_act:
            self.final_act = activation_dict[final_act]
        if meta_data:
            self.meta_merger = MergingModel(meta_data["method"], meta_data["params"])
        if squashed_embedding:
            self.squashed = torch.nn.Linear(seq_length, 1)
            self.unsquashed = torch.nn.Linear(1, seq_length)

    def make_embedding(self, x: torch.Tensor):
        """
        Creates the initial embedding, applies positional encoding, and passes through the encoder.

        :param x: The input tensor of shape (B, L, N).
        :type x: torch.Tensor
        :return: The encoded tensor.
        :rtype: torch.Tensor
        """
        x = self.dense_shape(x)
        x = self.pe(x)
        # (L, B, N)
        x = x.permute(1, 0, 2)
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)
        else:
            # Allow no mask
            x = self.transformer_enc(x)
        if self.squashed:
            x = x.permute(1, 2, 0)
            x = self.squashed(x)
        return x

    def __squashed__embedding(self, x: torch.Tensor):
        """
        Applies the squashing and unsquashing linear layers for the time dimension.

        :param x: The input tensor, typically after the encoder, of shape (L, B, N).
        :type x: torch.Tensor
        :return: The transformed tensor.
        :rtype: torch.Tensor
        """
        x = x.permute(1, 2, 0)  # (B, N, L)
        x = self.squashed(x)
        x = self.unsquashed(x)
        x = x.permute(0, 2, 1)  # (B, L, N)
        x = x.permute(1, 0, 2)  # (L, B, N)
        return x

    def forward(self, x: torch.Tensor, meta_data=None) -> torch.Tensor:
        """Performs forward pass on tensor of (batch_size, sequence_length, n_time_series) Return tensor of dim
        (batch_size, output_seq_length * output_dim) or (batch_size, output_seq_length, output_dim) if output_dim > 1.

        :param x: The input tensor of shape (B, L, N), where B is batch size, L is sequence length, and N is number of time series.
        :type x: torch.Tensor
        :param meta_data: Optional static meta-data tensor for the `meta_merger`.
        :type meta_data: torch.Tensor or None
        :return: The forecasted time series tensor.
        :rtype: torch.Tensor
        """
        x = self.dense_shape(x)
        if type(meta_data) == torch.Tensor:
            # batch_size = x.shape[0]
            # meta_data = meta_data.repeat(batch_size, 1).unsqueeze(2)
            # x = x.permute(0, 2, 1).contiguous()
            x = self.meta_merger(x, meta_data)
            # x = x.permute(0, 2, 1)
        x = self.pe(x)
        # (L, B, N)
        x = x.permute(1, 0, 2)
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)
        else:
            # Allow no mask
            x = self.transformer_enc(x)
        if self.squashed:
            x = self.__squashed__embedding(x)
        x = self.output_dim_layer(x)
        # (B, N, L)
        x = x.permute(1, 2, 0)
        x = self.out_length_lay(x)
        if self.final_act:
            x = self.final_act(x)
        if self.out_dim > 1:
            return x.permute(0, 2, 1)
        return x.view(-1, self.output_seq_length)


class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializes the SimplePositionalEncoding module.

        :param d_model: The embedding dimension of the model.
        :type d_model: int
        :param dropout: The dropout value to apply.
        :type dropout: float
        :param max_len: The maximum sequence length for which to pre-compute positional encodings.
        :type max_len: int
        """
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
        """Creates a basic positional encoding.

        :param x: The input tensor to which positional encoding is added.
        :type x: torch.Tensor
        :return: The input tensor with positional encoding added and dropout applied.
        :rtype: torch.Tensor
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def greedy_decode(
        model,
        src: torch.Tensor,
        max_len: int,
        real_target: torch.Tensor,
        unsqueeze_dim=1,
        output_len=1,
        device='cpu',
        multi_targets=1,
        probabilistic=False,
        scaler=None):
    """Mechanism to sequentially decode the model.

    :param model: The SimpleTransformer model instance.
    :param src: The Historical time series values (input to the encoder).
    :type src: torch.Tensor
    :param max_len: The maximum length of the sequence to decode.
    :type max_len: int
    :param real_target: The real values (they should be masked), however if you want can include known real values. This tensor is modified in place.
    :type real_target: torch.Tensor
    :param unsqueeze_dim: The dimension to unsqueeze the initial target vector `ys`. Defaults to 1.
    :type unsqueeze_dim: int
    :param output_len: Currently not used, defaults to 1.
    :type output_len: int
    :param device: The device on which the tensors should reside. Defaults to 'cpu'.
    :type device: str
    :param multi_targets: Currently not used, defaults to 1.
    :type multi_targets: int
    :param probabilistic: Currently not used, defaults to False.
    :type probabilistic: bool
    :param scaler: Currently not used, defaults to None.
    :type scaler: any
    :return: The decoded sequence of forecasted values.
    :rtype: torch.Tensor
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