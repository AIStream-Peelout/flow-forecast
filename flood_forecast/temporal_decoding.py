
import torch


def decoding_function(model, src: torch.Tensor, trg: torch.Tensor, forecast_length: int, src_temp: torch.Tensor,
                      tar_temp: torch.Tensor, unknown_cols_st: int, decoder_seq_len: int, max_len: int):
    """This function is responsible for decoding models that use `TemporalLoader` data. The basic logic of this
    function is as follows. The data to the encoder (e.g. src) is not modified at each step of the decoding process.
    Instead only the data to the decoder (e.g. the masked trg) is changed when forecasting max_len > forecast_length.
    New data is appended (forecast_len == 2) (decoder_seq==10) (max==20) (20 (8)->2 First 8 should

    :param model: The PyTorch time series forecasting model for
    :type model: `torch.nn.Module`
    :param src: The forecast_history tensor. Should be of dimension (batch_size, forecast_history, n_time_series).
    Ocassionally batch_size will not be present so at some points it will only be (forecast_history, n_time_series)
    :type src: torch.Tensor
    :param trg: The target tensor. Should be of dimension (batch_size, hours_to_forecast, n_time_series)
    :type trg: torch.Tensor
    :param forecast_length: The of leng th of the forecast the model makes at each forward pass.
    :type forecast_length: int
    :param src_temp: The
    :type src_temp: [type]
    :param tar_temp: The target's temporal feats. This should have a shape of (batch_size, max_len+diff, n_time_series)
    :type tar_temp: torch.Tensor
    :param unknown_cols_st: The un
    :type unknown_cols_st: int
    :param decoder_seq_len: The number of time steps passed into the decoder
    :type decoder_seq_len: int
    :param max_len: The number of time steps that should be in the final returned vector.
    :type max_len: int
    :return: [description]
    :rtype: [type]
    """
    if len(src.shape) == 2:
        # We assume batch_size is missing in this case
        # this should be ubiquitous
        src = src.unsqueeze(0)
        trg = trg.unsqueeze(0)
        src_temp = src_temp.unsqueeze(0)
        tar_temp = tar_temp.unsqueeze(0)
    out1 = torch.zeros_like(trg)
    filled_target = trg.clone()[:, 0:decoder_seq_len, :]
    filled_target[:, -forecast_length:, :] = torch.zeros_like(filled_target[:, -forecast_length:, :])
    # Useless variable to avoid long line error
    d = decoder_seq_len
    assert filled_target[:, -forecast_length:, :].any() != trg[:, d - forecast_length:decoder_seq_len, :].any()
    assert filled_target[0, -forecast_length, 0] != trg[0, -forecast_length, 0]
    for i in range(0, max_len, forecast_length):
        residual = decoder_seq_len if i + decoder_seq_len <= max_len else max_len % decoder_seq_len
        filled_target = filled_target[:, -residual:, :]
        if residual != decoder_seq_len:
            print("Shape of tar_temp resid")
            print(tar_temp[:, -residual:, :].shape)
            out = model(src, src_temp, filled_target, tar_temp[:, -residual:, :])
        else:
            print("Shape of tar_temp")
            print(tar_temp[:, -residual:, :].shape)
            out = model(src, src_temp, filled_target, tar_temp[:, i:i + residual, :])
        print("out shape is ")
        print(out.shape)
        residual1 = forecast_length if i + forecast_length <= max_len else max_len % forecast_length
        print("residual1 is")
        out1[:, i: i + residual1, :] = out[:, -residual1:, :]
        filled_target1 = torch.zeros_like(filled_target[:, 0:forecast_length * 2, :])
        print(filled_target1.shape[1])
        assert filled_target1.shape[1] == forecast_length * 2
        filled_target1[:, -forecast_length * 2:-forecast_length, :] = out[:, -forecast_length:, :]
        filled_target = torch.cat((filled_target, filled_target1), dim=1)
        print("Out shape below")
        print(filled_target.shape)
        assert out1[0, 0, 0] != 0
        assert out1[0, 0, 0] != 0
    return out1
