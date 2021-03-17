
import torch


def decoding_function(model, src: torch.Tensor, trg: torch.Tensor, forecast_length, src_temp,
                      tar_temp, unknown_cols_st: int, decoder_seq_len: int, max_len: int):
    """[summary]

    :param model: [description]
    :type model: [type]
    :param src: [description]
    :type src: torch.Tensor
    :param trg: [description]
    :type trg: torch.Tensor
    :param forecast_length: [description]
    :type forecast_length: [type]
    :param src_temp: [description]
    :type src_temp: [type]
    :param tar_temp: [description]
    :type tar_temp: [type]
    :param unknown_cols_st: [description]
    :type unknown_cols_st: int
    :param decoder_seq_len: [description]
    :type decoder_seq_len: int
    :param max_len: [description]
    :type max_len: int
    :return: [description]
    :rtype: [type]
    """
    out1 = torch.zeros_like(trg)
    filled_target = trg.clone()[:, 0:decoder_seq_len, :]
    filled_target[:, -forecast_length:, :] = torch.zeros_like(filled_target[:, -forecast_length:, :])
    for i in range(0, max_len, forecast_length):
        residual = decoder_seq_len if i + decoder_seq_len < max_len else max_len % decoder_seq_len
        filled_target = filled_target[:, -residual:, :]
        if residual != decoder_seq_len:
            out = model(src, src_temp, filled_target, tar_temp[:, -residual:, :])
        else:
            out = model(src, src_temp, filled_target, tar_temp[:, i:i + residual, :])
        residual1 = forecast_length if i + forecast_length < max_len else max_len % forecast_length
        out1[:, i: i + residual, :] = out[:, -residual1:, :]
        print("out shape is ")
        print(out.shape)
        filled_target1 = torch.zeros_like(filled_target[:, 0:forecast_length * 2, :])
        filled_target1[:, -forecast_length * 2:-forecast_length, :] = out[:, -forecast_length:, :]
        filled_target = torch.cat((filled_target, filled_target1), dim=1)
        print("Out shape below")
        print(filled_target.shape)
    return residual1
