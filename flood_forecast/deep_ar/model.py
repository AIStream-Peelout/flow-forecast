
import torch
import torch.nn as nn
from torch.autograd import Variable


class DeepAR(nn.Module):
    def __init__(self,
                 num_class: int,
                 cov_dim: int,
                 lstm_dropout: float,
                 embedding_dim: int,
                 lstm_hidden_dim: int,
                 lstm_layers: int,
                 sample_times: int,
                 predict_steps: int,
                 predict_start: int
                 ):
        """Initialize the DeepAR model.

        :param num_class: Number of classes
        :param cov_dim: Number of covariates
        :param lstm_dropout: drop out rate
        :param embedding_dim: dimension of embedding layer
        :param lstm_hidden_dim: hidden dimension of LSTM
        :param lstm_layers: Number of LSTM layers
        :param sample_times: sample time steps
        :param predict_steps: Number of steps to predict
        :param predict_start: Step to start prediction at
        """
        super(DeepAR, self).__init__()
        self.params = {}

        self.params["num_class"] = num_class
        self.params["cov_dim"] = cov_dim
        self.params["lstm_dropout"] = lstm_dropout
        self.params["embedding_dim"] = embedding_dim
        self.params["lstm_hidden_dim"] = lstm_hidden_dim
        self.params["lstm_layers"] = lstm_layers
        self.params["sample_times"] = sample_times
        self.params["predict_steps"] = predict_steps
        self.params["predict_start"] = predict_start
        # self.params = params
        self.embedding = nn.Embedding(self.params["num_class"], self.params["embedding_dim"])

        self.lstm = nn.LSTM(input_size=1 + self.params["cov_dim"] + self.params["embedding_dim"],
                            hidden_size=self.params["lstm_hidden_dim"],
                            num_layers=self.params["lstm_layers"],
                            bias=True,
                            batch_first=False,
                            dropout=self.params["lstm_dropout"])
        '''self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)'''
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(self.params["lstm_hidden_dim"] * self.params["lstm_layers"], 1)
        self.distribution_presigma = nn.Linear(self.params["lstm_hidden_dim"] * self.params["lstm_layers"], 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([predict_start+predict_steps, batch_size, cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            y: ([predict_start+predict_steps,batch_size,1]): will use z_{t-1} for predicting z_{t}, note: z_0 = 0
        Returns:
            mu ([predict_start+predict_steps,batch_size]): estimated mean of z_t for all time steps
            sigma ([predict_start+predict_steps,batch_size]): estimated standard deviation of z_t for all time steps
        '''
        hidden = self.init_hidden(x.shape[1])  # input batch size
        cell = self.init_cell(x.shape[1])  # input batch size
        z_0 = self.init_z0(x.shape[1])
        mu_concat = torch.Tensor([])
        mu = torch.Tensor([])
        sigma_concat = torch.Tensor([])
        for idx in range(x.shape[0]):
            # onehot_embed = self.embedding(idx)
            # lstm_input = torch.cat((x, onehot_embed), dim=2)
            i = x[idx:idx + 1, :, :]
            if idx == 0:  # initial step
                z = z_0
            elif idx < self.params["predict_start"]:  # training period
                z = y[idx - 1: idx, :, :]
            else:  # prediction period
                z = mu.unsqueeze(0)
            # print(mu.shape)
            lstm_input = torch.cat((i, z), dim=2)
            # print(idx,lstm_input.shape)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            # use h from all three layers to calculate mu and sigma
            hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
            pre_sigma = self.distribution_presigma(hidden_permute)
            mu = self.distribution_mu(hidden_permute)
            sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive
            mu_concat = torch.cat((mu_concat, mu.unsqueeze(0)))
            sigma_concat = torch.cat((sigma_concat, sigma.unsqueeze(0)))
        return mu_concat, sigma_concat

    def init_hidden(self, input_size):
        return torch.zeros(self.params["lstm_layers"], input_size, self.params["lstm_hidden_dim"])

    def init_cell(self, input_size):
        return torch.zeros(self.params["lstm_layers"], input_size, self.params["lstm_hidden_dim"])

    def test(self, x, v_batch, id_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        if sampling:
            samples = torch.zeros(self.params["sample_times"], batch_size, self.params["predict_steps"])
            for j in range(self.params["sample_times"]):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params["predict_steps"]):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(
                        x[self.params["predict_start"] + t].unsqueeze(0),
                        id_batch, decoder_hidden, decoder_cell)
                    gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                    pred = gaussian.sample()  # not scaled
                    samples[j, :, t] = pred * v_batch[:, 0] + v_batch[:, 1]
                    if t < (self.params["predict_steps"] - 1):
                        x[self.params["predict_start"] + t + 1, :, 0] = pred

            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma

        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(batch_size, self.params["predict_steps"])
            sample_sigma = torch.zeros(batch_size, self.params["predict_steps"])
            for t in range(self.params["predict_steps"]):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params["predict_start"] + t].unsqueeze(0),
                                                                     id_batch, decoder_hidden, decoder_cell)
                sample_mu[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
                sample_sigma[:, t] = sigma_de * v_batch[:, 0]
                if t < (self.params["predict_steps"] - 1):
                    x[self.params["predict_start"] + t + 1, :, 0] = mu_de
            return sample_mu, sample_sigma


def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    zero_index = (labels != 0)
    distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
    likelihood = distribution.log_prob(labels[zero_index])
    return -torch.mean(likelihood)
