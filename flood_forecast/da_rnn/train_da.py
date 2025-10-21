import typing
from typing import Tuple
import json
import os

import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

import numpy as np
from flood_forecast.da_rnn import utils
from flood_forecast.da_rnn.constants import device
from flood_forecast.da_rnn.modules import Encoder, Decoder
from flood_forecast.da_rnn.custom_types import DaRnnNet, TrainData, TrainConfig
from flood_forecast.da_rnn.utils import numpy_to_tvar
from torch.utils.tensorboard import SummaryWriter


logger = utils.setup_log()
logger.info(f"Using computation device: {device}")


def da_rnn(train_data: TrainData,
           n_targs: int,
           encoder_hidden_size=64,
           decoder_hidden_size=64,
           T=10,
           learning_rate=0.01,
           batch_size=128,
           param_output_path="",
           save_path: str = "") -> Tuple[dict, DaRnnNet]:
    """
    Initializes and loads the Data Attention Recurrent Neural Network (DA-RNN) model,
    its optimizers, and the training configuration.

    :param train_data: Contains the features and targets for training and validation.
    :type train_data: flood_forecast.da_rnn.custom_types.TrainData
    :param n_targs: The number of target columns (not steps).
    :type n_targs: int
    :param encoder_hidden_size: The size of the hidden state in the encoder GRU.
    :type encoder_hidden_size: int
    :param decoder_hidden_size: The size of the hidden state in the decoder GRU.
    :type decoder_hidden_size: int
    :param T: The number of timesteps in the look-back window.
    :type T: int
    :param learning_rate: The initial learning rate for the Adam optimizers.
    :type learning_rate: float
    :param batch_size: The batch size used for training.
    :type batch_size: int
    :param param_output_path: Directory path to save the encoder and decoder argument JSON files.
    :type param_output_path: str
    :param save_path: Path to a directory containing previously saved 'encoder.pth' and 'decoder.pth' files for resuming training.
    :type save_path: str
    :return: A tuple containing the training configuration and the initialized DA-RNN network wrapper.
    :rtype: typing.Tuple[dict, flood_forecast.da_rnn.custom_types.DaRnnNet]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    enc_kwargs = {
        "input_size": train_data.feats.shape[1],
        "hidden_size": encoder_hidden_size,
        "T": T}
    encoder = Encoder(**enc_kwargs).to(device)
    with open(os.path.join(param_output_path, "enc_kwargs.json"), "w+") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    decoder = Decoder(**dec_kwargs).to(device)
    with open(os.path.join(param_output_path, "dec_kwargs.json"), "w+") as fi:
        json.dump(dec_kwargs, fi, indent=4)
    if save_path:
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        print("Resuming training from " + os.path.join(save_path, "encoder.pth"))
        encoder.load_state_dict(
            torch.load(
                os.path.join(
                    save_path,
                    "encoder.pth"),
                map_location=device))
        decoder.load_state_dict(
            torch.load(
                os.path.join(
                    save_path,
                    "decoder.pth"),
                map_location=device))

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)
    return train_cfg, da_rnn_net


def train(
        net: DaRnnNet,
        train_data: TrainData,
        t_cfg: TrainConfig,
        train_config="",
        n_epochs=10,
        save_plots=True,
        wandb=False,
        tensorboard=False):
    """
    The main training loop for the DA-RNN model.

    :param net: The initialized DA-RNN network wrapper containing the encoder, decoder, and optimizers.
    :type net: flood_forecast.da_rnn.custom_types.DaRnnNet
    :param train_data: The dataset container with features and targets.
    :type train_data: flood_forecast.da_rnn.custom_types.TrainData
    :param t_cfg: The training configuration object containing constants like T, batch size, and loss function.
    :type t_cfg: flood_forecast.da_rnn.custom_types.TrainConfig
    :param train_config: Unused placeholder.
    :type train_config: str
    :param n_epochs: The number of epochs to train the model for.
    :type n_epochs: int
    :param save_plots: If True, save the prediction plots to disk; otherwise, display them.
    :type save_plots: bool
    :param wandb: If True, log metrics and plots to Weights & Biases.
    :type wandb: bool
    :param tensorboard: If True, log metrics to TensorBoard.
    :type tensorboard: bool
    :return: A tuple containing a list of loss history ([iter_losses, epoch_losses]) and the trained network object.
    :rtype: typing.Tuple[typing.List[numpy.ndarray, numpy.ndarray], flood_forecast.da_rnn.custom_types.DaRnnNet]
    """
    if wandb:
        import wandb
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(
        f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")
    n_iter = 0
    if tensorboard:
        writer = SummaryWriter()

    for e_i in range(n_epochs):
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            feats, y_history, y_target = prep_train_data(batch_idx, t_cfg, train_data)
            if len(feats) > 0 and len(y_target) > 0 and len(y_history) > 0:
                loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
                iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
                n_iter += 1
                adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(
            iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        if e_i % 1 == 0:
            y_test_pred = predict(net, train_data,
                                  t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                  on_train=False)
            # TODO: make this MSE and make it work for multiple inputs
            val_loss = y_test_pred - train_data.targs[t_cfg.train_size:]
            logger.info(
                f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}.")
            y_train_pred = predict(net, train_data,
                                   t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                   on_train=True)

            # train_mse = np.mean((y_train_pred-train_data.targs[:t_cfg.train_size])**2)
            mse = np.mean((y_test_pred - train_data.targs[t_cfg.train_size:])**2)
            if wandb:
                wandb.log({"epoch": n_epochs, "validation_loss": val_loss, "validation_mse": mse})
            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
                     label="True")
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
                     label='Predicted - Train')
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
                     label='Predicted - Test')
            plt.legend(loc='upper left')
            utils.save_or_show_plot(f"pred_{e_i}.png", save_plots)
            if tensorboard:
                # writer.add_scalar('Loss/Validation', val_loss, e_i)
                writer.add_scalar('Validation/MSE', mse, e_i)  # Check MSE CALC
                # writer.add_scalar("Train/MSE", train_mse, e_i )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
    print(os.path.join(dir_path, "checkpoint", "encoder.pth"))
    torch.save(net.encoder.state_dict(), os.path.join(dir_path, "checkpoint", "encoder.pth"))
    torch.save(net.decoder.state_dict(), os.path.join(dir_path, "checkpoint", "decoder.pth"))

    return [iter_losses, epoch_losses], net


def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData) -> Tuple:
    """
    Prepares a batch of training data by slicing the features (X), historical targets (y_history),
    and target (y_target) based on the given batch indices and look-back window (T).

    :param batch_idx: A numpy array of indices indicating the start point of each sequence in the batch.
    :type batch_idx: numpy.ndarray
    :param t_cfg: The training configuration object containing constants like T.
    :type t_cfg: flood_forecast.da_rnn.custom_types.TrainConfig
    :param train_data: The dataset container with features and targets.
    :type train_data: flood_forecast.da_rnn.custom_types.TrainData
    :return: A tuple containing the batched features (X), historical targets (y_history), and true targets (y_target).
    :rtype: typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet, n_iter: int) -> None:
    """
    Adjusts the learning rate of the encoder and decoder optimizers based on the number of iterations.
    The learning rate is reduced by 10% every 10,000 iterations.

    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts
    # https://www.jeremyjordan.me/nn-learning-rate/

    :param net: The DA-RNN network wrapper.
    :type net: flood_forecast.da_rnn.custom_types.DaRnnNet
    :param n_iter: The current total number of training iterations.
    :type n_iter: int
    :return: None
    :rtype: None
    """
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts
    # https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9


def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target) -> float:
    """
    Performs a single training step: forward pass, loss calculation, backward pass, and optimizer steps.

    :param t_net: The DA-RNN network wrapper.
    :type t_net: flood_forecast.da_rnn.custom_types.DaRnnNet
    :param loss_func: The loss function to use for calculating error (e.g., nn.MSELoss).
    :type loss_func: typing.Callable
    :param X: The input features batch (numpy array).
    :type X: numpy.ndarray
    :param y_history: The historical targets batch (numpy array).
    :type y_history: numpy.ndarray
    :param y_target: The true target values batch (numpy array).
    :type y_target: numpy.ndarray
    :return: The calculated loss value for the current iteration.
    :rtype: float
    """
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()
    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))

    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item()


def predict(
        t_net: DaRnnNet,
        t_dat: TrainData,
        train_size: int,
        batch_size: int,
        T: int,
        on_train=False) -> np.ndarray:
    """
    Generates predictions for either the training set or the validation/test set.

    :param t_net: The trained DA-RNN network wrapper.
    :type t_net: flood_forecast.da_rnn.custom_types.DaRnnNet
    :param t_dat: The full dataset container with features and targets.
    :type t_dat: flood_forecast.da_rnn.custom_types.TrainData
    :param train_size: The number of data points used for training. Determines the split point for predictions.
    :type train_size: int
    :param batch_size: The batch size to use during prediction.
    :type batch_size: int
    :param T: The number of timesteps in the look-back window.
    :type T: int
    :param on_train: If True, calculate predictions for the training set; otherwise, calculate for the validation/test set.
    :type on_train: bool
    :return: A numpy array of predicted target values.
    :rtype: numpy.ndarray
    """
    out_size = t_dat.targs.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred
