from torch import nn
import torch 
import math
from torch.nn.modules import Transformer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer, LayerNorm
from torch.autograd import Variable 
class SimpleTransformer(torch.nn.Module):
    def __init__(self, number_time_series, seq_length=48, output_seq_len = None, d_model=128, n_heads=8):
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

    def forward(self, x, t, tgt_mask=None, src_mask=None):
        if src_mask:
            x = self.encode_sequence(x, src_mask)
        else: 
            x = self.encode_sequence(x, src_mask)
        return self.decode_seq(x, t, tgt_mask)
    
    def basic_feature(self, x):
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        return x 
    
    def encode_sequence(self, x, src_mask=None):
        x = self.basic_feature(x)
        x = self.transformer.encoder(x, src_mask)
        return x
    
    def decode_seq(self, mem, t, tgt_mask=None, view_number=None):
        if view_number is None:
            view_number = self.out_seq_len
        if tgt_mask is None:
            tgt_mask = self.tgt_mask
        t = self.basic_feature(t)
        x = self.transformer.decoder(t, mem, tgt_mask=tgt_mask)
        x = self.final_layer(x)
        return x.view(-1, view_number)
    
class CustomTransformer(torch.nn.Module):
    def __init__(self, n_time_series, d_model=128):
        super().__init__()
        self.dense_shape = torch.nn.Linear(n_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, 8)
        encoder_norm = LayerNorm(d_model)
        self.transformer_enc = TransformerEncoder(encoder_layer, 6, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, 8, 2048, 0.1)
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.final_layer = torch.nn.Linear(d_model, 1)
    def forward(self, x, t, tgt_mask):
        x = self.dense_shape(x)
        x = self.pe(x)
        t = self.dense_shape(t)
        t = self.pe(t)
        x = x.permute(1,0,2)
        t = t.permute(1,0,2)
        x = self.transformer_enc(x, tgt_mask)
        x = self.transformer_decoder(x, t, tgt_mask)
        x = self.final_layer(x)
        return x
    
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

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
def generate_square_subsequent_mask(sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
def greedy_decode(model, src, max_len, real_target, start_symbol, unsqueeze_dim=1, device='cpu'):
    """
    Mechanism to sequentially decode the model
    """
    src_mask = model.mask
    memory = model.encode_sequence(src, src_mask)
    ys = start_symbol[:, -1, :].unsqueeze(unsqueeze_dim)
    for i in range(max_len-1):
        mask = generate_square_subsequent_mask(i+1).to(device)
        with torch.no_grad():
            print(memory.shape)
            print(ys.shape)
            print(mask.shape)
            out = model.decode_seq(memory, 
                               Variable(ys), 
                              Variable(mask), i+1)
            real_target[:, i, 0] = out[:, i]
            ys = torch.cat((ys, real_target[:, i, :].unsqueeze(1)), 1)
    return ys