import torch
import torch.nn as nn


class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, forecast_history: int, forecast_length: int, enc_in=128, individual=False, n_targs=1):
        super(NLinear, self).__init__()
        self.seq_len = forecast_history
        self.pred_len2 = forecast_length
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = enc_in
        self.individual = individual
        self.n_targs = n_targs
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len2))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len2, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                print(output.shape)
                print('shpae is ')
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        if self.n_targs == 1:
            return x[:, :, -1]
        return x  # [Batch, Output length, Channel]


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, forecast_history: int, forecast_length: int, individual, enc_in: int, n_targs=1):
        """Code from

        :param forecast_history: _description_
        :type forecast_history: int
        :param forecast_length: _description_
        :type forecast_length: int
        :param individual: _description_
        :type individual: _type_
        :param enc_in: _description_
        :type enc_in: int
        """
        super(DLinear, self).__init__()
        self.seq_len = forecast_history
        self.pred_len2 = forecast_length
        self.n_targs = n_targs

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len2))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len2))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len])
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len2)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len2)
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len2,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len2,self.seq_len]))

    def forward(self, x: torch.Tensor):
        """The

        :param x: PyTorch tensor of size [Batch, Input length, Channel]
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),
                                          seasonal_init.size(1), self.pred_len2],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1),
                                        self.pred_len2], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Badtch, Output length, Channel]
        if self.n_targs == 1:
            return x[:, :, -1]
        else:
            return x
