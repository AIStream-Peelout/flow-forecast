
import torch
from typing import List


def decoding_function(model, src: torch.Tensor, trg: torch.Tensor, forecast_length, src_temp,
                      tar_temp, unknown_cols_st: int, decoder_seq_len: int):
    filled_target = trg.clone()[:, 0:decoder_seq_len, :]
    filled_target[:, -forecast_length:, :] = torch.zeros_like(filled_target[:, -forecast_length:, :])
    for i in range(0, 336, forecast_length):
        residual = decoder_seq_len if i + decoder_seq_len < 336 else 336 % decoder_seq_len
        filled_target = filled_target[:, -residual:, :]
        out = model(src, src_temp, filled_target, tar_temp[:, i:i + residual, :])
        print("out shape is ")
        print(out.shape)
        filled_target1 = torch.zeros_like(filled_target[:, 0:forecast_length * 2, :])
        filled_target1[:, -forecast_length * 2:-forecast_length, :] = out[:, -forecast_length:, :]
        filled_target = torch.cat((filled_target, filled_target1), dim=1)
        print("Out shape below")
        print(filled_target.shape)
