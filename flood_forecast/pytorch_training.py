import torch
import torch.optim as optim
from typing import Type, Dict, List, Union
from torch.utils.data import DataLoader
import json
import wandb
from flood_forecast.utils import numpy_to_tvar
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.model_dict_function import pytorch_opt_dict, pytorch_criterion_dict
from flood_forecast.transformer_xl.transformer_basic import greedy_decode
from flood_forecast.basic.linear_regression import simple_decode
from flood_forecast.training_utils import EarlyStopper
from flood_forecast.custom.custom_opt import GaussianLoss, MASELoss
from torch.nn import CrossEntropyLoss


def multi_crit(crit_multi: List, output, labels, valid=None):
    i = 0
    loss = 0.0
    for crit in crit_multi:
        if len(output.shape) == 3:
            loss += compute_loss(labels[:, :, i], output[:, :, i], torch.rand(1, 2), crit, valid)
        else:
            loss += compute_loss(labels[:, i], output[:, i], torch.rand(1, 2), crit, valid)
    summed_loss = loss
    return summed_loss


def handle_meta_data(model: PyTorchForecast):
    """A function to initialize models with meta-data
    :param model: A PyTorchForecast model with meta_data parameter block in config file.
    :type model: PyTorchForecast
    :return: Returns a tuple of the initial meta-representation
    :rtype: tuple(PyTorchForecast, torch.Tensor, float)
    """
    meta_loss = None
    with open(model.params["meta_data"]["path"]) as f:
        json_data = json.load(f)
    if "meta_loss" in model.params["meta_data"]:
        meta_loss_str = model.params["meta_data"]["meta_loss"]
        meta_loss = pytorch_criterion_dict[meta_loss_str]()
    dataset_params2 = json_data["dataset_params"]
    training_path = dataset_params2["training_path"]
    valid_path = dataset_params2["validation_path"]
    meta_name = json_data["model_name"]
    meta_model = PyTorchForecast(meta_name, training_path, valid_path, dataset_params2["test_path"], json_data)
    meta_representation = get_meta_representation(model.params["meta_data"]["column_id"],
                                                  model.params["meta_data"]["uuid"], meta_model)
    return meta_model, meta_representation, meta_loss


def make_crit(model_params: Dict) -> Union[torch.nn.Module, List]:
    """A function to create the criterion for training from the parameters
    :param model_params: The training params Dict block in FF
    :type model_params: Dict
    """
    training_params = model_params
    criterion_init_params = {}
    if "criterion_params" in training_params:
        criterion_init_params = training_params["criterion_params"]
    if type(training_params["criterion"]) == list:
        criterion = []
        i = 0
        for crit, param in zip(training_params["criterion"], criterion_init_params):
            res = pytorch_criterion_dict[crit](**param)
            i += 1
            criterion.append(res)
    else:
        criterion = pytorch_criterion_dict[training_params["criterion"]](**criterion_init_params)
    return criterion


def train_transformer_style(
        model: PyTorchForecast,
        training_params: Dict,
        takes_target=False,
        forward_params: Dict = {},
        model_filepath: str = "model_save",
        class2=False) -> None:

    """Function to train any PyTorchForecast model

    :param model:  A properly wrapped PyTorchForecast model
    :type model: PyTorchForecast
    :param training_params: A dictionary of the necessary parameters for training.
    :type training_params: Dict
    :param takes_target: A parameter to determine whether a model requires the target, defaults to False
    :type takes_target: bool, optional
    :param forward_params: [description], defaults to {}
    :type forward_params: Dict, optional
    :param model_filepath: The file path to load modeel weights from, defaults to "model_save"
    :type model_filepath: str, optional
    :raises ValueError: Has an error
    """
    use_wandb = model.wandb
    es = None
    worker_num = 1
    pin_memory = False
    dataset_params = model.params["dataset_params"]
    num_targets = 1
    if "n_targets" in model.params:
        num_targets = model.params["n_targets"]
    if "num_workers" in dataset_params:
        worker_num = dataset_params["num_workers"]
        print("using " + str(worker_num))
    if "pin_memory" in dataset_params:
        pin_memory = dataset_params["pin_memory"]
        print("Pin memory set to true")
    if "early_stopping" in model.params:
        es = EarlyStopper(model.params["early_stopping"]['patience'])
    if "shuffle" not in training_params:
        training_params["shuffle"] = False
    criterion = make_crit(training_params)
    opt = pytorch_opt_dict[training_params["optimizer"]](
        model.model.parameters(), **training_params["optim_params"])
    if "probabilistic" in model.params["model_params"] or "probabilistic" in model.params:
        probabilistic = True
    else:
        probabilistic = False
    max_epochs = training_params["epochs"]
    data_loader = DataLoader(
        model.training,
        batch_size=training_params["batch_size"],
        shuffle=training_params["shuffle"],
        sampler=None,
        batch_sampler=None,
        num_workers=worker_num,
        collate_fn=None,
        pin_memory=pin_memory,
        drop_last=False,
        timeout=0,
        worker_init_fn=None)
    validation_data_loader = DataLoader(
        model.validation,
        batch_size=training_params["batch_size"],
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=worker_num,
        collate_fn=None,
        pin_memory=pin_memory,
        drop_last=False,
        timeout=0,
        worker_init_fn=None)
    # TODO support batch_size > 1
    test_data_loader = DataLoader(model.test_data, batch_size=1, shuffle=False, sampler=None,
                                  batch_sampler=None, num_workers=worker_num, collate_fn=None,
                                  pin_memory=pin_memory, drop_last=False, timeout=0,
                                  worker_init_fn=None)
    meta_model = None
    meta_representation = None
    meta_loss = None
    if model.params.get("meta_data") is not None:
        # model.params["meta_data"] = False
        meta_model, meta_representation, meta_loss = handle_meta_data(model)
    if use_wandb:
        wandb.watch(model.model)
    use_decoder = False
    if "use_decoder" in model.params:
        use_decoder = True
    session_params = []
    for epoch in range(max_epochs):
        total_loss = torch_single_train(
            model,
            opt,
            criterion,
            data_loader,  # s
            takes_target,
            meta_model,
            meta_representation,
            meta_loss,
            multi_targets=num_targets)
        print("The loss for epoch " + str(epoch))
        print(total_loss)
        valid = compute_validation(
            validation_data_loader,
            model.model,
            epoch,
            model.params["dataset_params"]["forecast_length"],
            model.crit,
            model.device,
            multi_targets=num_targets,
            meta_model=meta_model,
            decoder_structure=use_decoder,
            use_wandb=use_wandb,
            probabilistic=probabilistic,
            classification=class2)
        if valid == 0.0:
            raise ValueError("Error validation loss is zero there is a problem with the validator.")
        if use_wandb:
            wandb.log({'epoch': epoch, 'loss': total_loss})
        epoch_params = {
            "epoch": epoch,
            "train_loss": str(total_loss),
            "validation_loss": str(valid)}
        session_params.append(epoch_params)
        if es:
            if not es.check_loss(model.model, valid):
                print("Stopping model now")
                model.model.load_state_dict(torch.load("checkpoint.pth"))
                break
    decoder_structure = True
    the_ae = model.params["dataset_params"]["class"] == "AutoEncoder"
    if the_ae or model.params["dataset_params"]["class"] == "GeneralClassificationLoader":
        decoder_structure = False
    test = compute_validation(
        test_data_loader,
        model.model,
        epoch,
        model.params["dataset_params"]["forecast_length"],
        model.crit,
        model.device,
        meta_model=meta_model,
        multi_targets=num_targets,
        decoder_structure=decoder_structure,
        use_wandb=use_wandb,
        val_or_test="test_loss",
        probabilistic=probabilistic,
        classification=class2)
    print("test loss:", test)
    model.params["run"] = session_params
    model.save_model(model_filepath, max_epochs)


def get_meta_representation(column_id: str, uuid: str, meta_model: PyTorchForecast) -> torch.Tensor:
    return meta_model.test_data.__getitem__(0, uuid, column_id)[0]


def handle_scaling(validation_dataset, src, output: torch.Tensor, labels, probabilistic, m, output_std):
    """Function that handles un-scaling the model output.

    :param validation_dataset: A dataset object for the validation dataset. We use its inverse scale method.
    :type validation_dataset: [type]
    :param src: [description]
    :type src: torch.Tensor
    :param output: [description]
    :type output: torch.Tensor
    :param labels: [description]
    :type labels: torch.Tensor
    :param probabilistic: Whether the model is probablisitic or not.
    :type probabilistic: bool
    :param m: Whether there are multiple targets
    :type m: int
    :param output_std: [description]
    :type output_std: [type]
    :return: [description]
    :rtype: [type]
    """
    # To-do move to class fun ction
    output_dist = None
    if probabilistic:
        unscaled_out = validation_dataset.inverse_scale(output)
        try:
            output_std = numpy_to_tvar(output_std)
        except Exception:
            pass
        output_dist = torch.distributions.Normal(unscaled_out, output_std)
    elif m > 1:
        output = validation_dataset.inverse_scale(output.cpu())
        labels = validation_dataset.inverse_scale(labels.cpu())
    elif len(output.shape) == 3:
        output = output.cpu().numpy().transpose(0, 2, 1)
        labels = labels.cpu().numpy().transpose(0, 2, 1)
        output = validation_dataset.inverse_scale(torch.from_numpy(output))
        labels = validation_dataset.inverse_scale(torch.from_numpy(labels))
        stuff = src.cpu().numpy().transpose(0, 2, 1)
        src = validation_dataset.inverse_scale(torch.from_numpy(stuff))
    else:
        output = validation_dataset.inverse_scale(output.cpu().transpose(1, 0))
        labels = validation_dataset.inverse_scale(labels.cpu().transpose(1, 0))
        src = validation_dataset.inverse_scale(src.cpu().transpose(1, 0))
    return src, output, labels, output_dist


def compute_loss(labels, output, src, criterion, validation_dataset, probabilistic=None, output_std=None, m=1):
    """Function for computing the loss

    :param labels: The real values for the target. Shape can be variable but should follow (batch_size, time)
    :type labels: torch.Tensor
    :param output: The output of the model
    :type output: torch.Tensor
    :param src: The source values (only really needed for the MASELoss function)
    :type src: torch.Tensor
    :param criterion: The loss function to use
    :type criterion: torch.nn.Loss or some variation
    :param validation_dataset: Only passed when unscaling of data is needed.
    :type validation_dataset: torch.utils.data.dataset
    :param probabilistic: Whether the model is a probabalistic returns a distribution, defaults to None
    :type probabilistic: bool, optional
    :param output_std: The standard distribution, defaults to None
    :type output_std: [type], optional
    :param m: The number of targets defaults to 1
    :type m: int, optional
    :return: Returns the computed loss
    :rtype: float
"""
    if isinstance(criterion, GaussianLoss):
        if len(output[0].shape) > 2:
            g_loss = GaussianLoss(output[0][:, :, 0], output[1][:, :, 0])
        else:
            g_loss = GaussianLoss(output[0][:, 0], output[1][:, 0])
        loss = g_loss(labels)
        return loss
    if not probabilistic and isinstance(output, torch.Tensor):
        if len(labels.shape) != len(output.shape):
            if len(labels.shape) > 1:
                if labels.shape[1] == output.shape[1]:
                    labels = labels.unsqueeze(2)
                else:
                    labels = labels.unsqueeze(0)
    if probabilistic:
        if type(output_std) != torch.Tensor:
            output_std = torch.from_numpy(output_std)
        if type(output) != torch.Tensor:
            output = torch.from_numpy(output)
        output_dist = torch.distributions.Normal(output, output_std)
    if validation_dataset:
        src, output, labels, output_dist = handle_scaling(validation_dataset, src, output, labels,
                                                          probabilistic, m, output_std)
    if probabilistic:
        loss = -output_dist.log_prob(labels.float()).sum()  # FIX THIS?
    elif isinstance(criterion, MASELoss):
        assert len(labels.shape) == len(output.shape)
        loss = criterion(labels.float(), output, src, m)
    elif isinstance(criterion, CrossEntropyLoss):
        if len(labels.shape) > 2:
            labels = labels.permute(0, 2, 1)
            output = output.permute(0, 2, 1)
        labels = labels.max(dim=1)[1]
        loss = criterion(output, labels)
    else:
        assert len(labels.shape) == len(output.shape)
        assert labels.shape[0] == output.shape[0]
        loss = criterion(output, labels.float())
    return loss


def torch_single_train(model: PyTorchForecast,
                       opt: optim.Optimizer,
                       criterion: Type[torch.nn.modules.loss._Loss],
                       data_loader: DataLoader,
                       takes_target: bool,
                       meta_data_model: PyTorchForecast,
                       meta_data_model_representation: torch.Tensor,
                       meta_loss=None,
                       multi_targets=1,
                       forward_params: Dict = {}) -> float:
    """Function that performs training of a single model. Runs through one epoch of the data.

    :param model: The PyTorchForecast model that is trained
    :type model: PyTorchForecast
    :param opt: The optimizer to use in the code
    :type opt: optim.Optimizer
    :param criterion: [description]
    :type criterion: Type[torch.nn.modules.loss._Loss]
    :param data_loader: [description]
    :type data_loader: DataLoader
    :param takes_target: A boolean that indicates whether the model takes the target during training
    :type takes_target: bool
    :param meta_data_model: If supplied a model that handles meta-data else None.
    :type meta_data_model: PyTorchForecast
    :param meta_data_model_representation: [description]
    :type meta_data_model_representation: torch.Tensor
    :param meta_loss: [description], defaults to None
    :type meta_loss: [type], optional
    :param multi_targets: [description], defaults to 1
    :type multi_targets: int, optional
    :param forward_params: [description], defaults to {}
    :type forward_params: Dict, optional
    :raises ValueError: [description]
    :return: [description]
    :rtype: float
    """

    probablistic = None
    if "probabilistic" in model.params["model_params"]:
        probablistic = True
    print('running torch_single_train')
    i = 0
    output_std = None
    mulit_targets_copy = multi_targets
    running_loss = 0.0
    for src, trg in data_loader:
        opt.zero_grad()
        if meta_data_model:
            representation = meta_data_model.model.generate_representation(meta_data_model_representation)
            forward_params["meta_data"] = representation
            if meta_loss:
                output = meta_data_model.model(meta_data_model_representation)
                met_loss = compute_loss(meta_data_model_representation, output, torch.rand(2, 3, 2), meta_loss, None)
                met_loss.backward()
        if takes_target:
            forward_params["t"] = trg
        elif "TemporalLoader" == model.params["dataset_params"]["class"]:
            forward_params["x_mark_enc"] = src[1].to(model.device)
            forward_params["x_dec"] = trg[1].to(model.device)
            forward_params["x_mark_dec"] = trg[0].to(model.device)
            src = src[0]
            pred_len = model.model.pred_len
            trg = trg[0]
            trg[:, -pred_len:, :] = torch.zeros_like(trg[:, -pred_len:, :].long()).float().to(model.device)
            # Assign to avoid other if statement
        elif "SeriesIDLoader" == model.params["dataset_params"]["class"]:
            pass
        src = src.to(model.device)
        trg = trg.to(model.device)
        output = model.model(src, **forward_params)
        if hasattr(model.model, "pred_len"):
            multi_targets = mulit_targets_copy
            pred_len = model.model.pred_len
            output = output[:, :, 0:multi_targets]
            labels = trg[:, -pred_len:, 0:multi_targets]
            multi_targets = False
        if model.params["dataset_params"]["class"] == "GeneralClassificationLoader":
            labels = trg
        elif multi_targets == 1:
            labels = trg[:, :, 0]
        elif multi_targets > 1:
            labels = trg[:, :, 0:multi_targets]
        if probablistic:
            output1 = output
            output = output.mean
            output_std = output1.stddev
        if type(criterion) == list:
            loss = multi_crit(criterion, output, labels, None)
        else:
            loss = compute_loss(labels, output, src, criterion, None, probablistic, output_std, m=multi_targets)
        if loss > 100:
            print("Warning: high loss detected")
        loss.backward()
        opt.step()
        if torch.isnan(loss) or loss == float('inf'):
            raise ValueError("Error infinite or NaN loss detected. Try normalizing data or performing interpolation")
        running_loss += loss.item()
        i += 1
    print("The running loss is: ")
    print(running_loss)
    print("The number of items in train is: " + str(i))
    total_loss = running_loss / float(i)
    return total_loss


def handle_crit_list():
    pass


def compute_validation(validation_loader: DataLoader,
                       model,
                       epoch: int,
                       sequence_size: int,
                       criterion: Type[torch.nn.modules.loss._Loss],
                       device: torch.device,
                       decoder_structure=False,
                       meta_data_model=None,
                       use_wandb: bool = False,
                       meta_model=None,
                       multi_targets=1,
                       val_or_test="validation_loss",
                       probabilistic=False,
                       classification=False) -> float:
    """Function to compute the validation loss metrics

    :param validation_loader: The data-loader of either validation or test-data
    :type validation_loader: DataLoader
    :param model: model
    :type model: [type]
    :param epoch: The epoch where the validation/test loss is being computed.
    :type epoch: int
    :param sequence_size: The length of the sequence (equivalent too
    :type sequence_size: int
    :param criterion: [description]
    :type criterion: Type[torch.nn.modules.loss._Loss]
    :param device: The device
    :type device: torch.device
    :param decoder_structure: Whether the model should use sequential decoding, defaults to False
    :type decoder_structure: bool, optional
    :param meta_data_model: The model to handle the meta-data, defaults to None
    :type meta_data_model: PyTorchForecast, optional
    :param use_wandb: Whether Weights and Biases is in use, defaults to False
    :type use_wandb: bool, optional
    :param meta_model: Whether the model leverages meta-data, defaults to None
    :type meta_model: bool, optional
    :param multi_targets: Whether the model, defaults to 1
    :type multi_targets: int, optional
    :param val_or_test: Whether validation or test loss is computed, defaults to "validation_loss"
    :type val_or_test: str, optional
    :param probabilistic: Whether the model is probablistic, defaults to False
    :type probabilistic: bool, optional
    :return: The loss of the first metric in the list.
    :rtype: float
    """
    print('Computing validation loss')
    unscaled_crit = dict.fromkeys(criterion, 0)
    scaled_crit = dict.fromkeys(criterion, 0)
    model.eval()
    output_std = None
    multi_targs1 = multi_targets
    scaler = None
    if validation_loader.dataset.no_scale:
        scaler = validation_loader.dataset
    with torch.no_grad():
        i = 0
        loss_unscaled_full = 0.0
        label_list = []
        mod_output_list = []
        for src, targ in validation_loader:
            src = src if isinstance(src, list) else src.to(device)
            targ = targ if isinstance(targ, list) else targ.to(device)
            # targ = targ if isinstance(targ, list) else targ.to(device)
            i += 1
            if decoder_structure:
                if type(model).__name__ == "SimpleTransformer":
                    targ_clone = targ.detach().clone()
                    output = greedy_decode(
                        model,
                        src,
                        targ.shape[1],
                        targ_clone,
                        device=device)[
                        :,
                        :,
                        0]
                elif type(model).__name__ == "Informer":
                    multi_targets = multi_targs1
                    filled_targ = targ[1].clone()
                    pred_len = model.pred_len
                    filled_targ[:, -pred_len:, :] = torch.zeros_like(filled_targ[:, -pred_len:, :]).float().to(device)
                    output = model(src[0].to(device), src[1].to(device), filled_targ.to(device), targ[0].to(device))
                    labels = targ[1][:, -pred_len:, 0:multi_targets].to(device)
                    src = src[0]
                    assert output.shape[1] != 0
                    assert labels.shape[1] != 0
                else:
                    output = simple_decode(model=model,
                                           src=src,
                                           max_seq_len=targ.shape[1],
                                           real_target=targ,
                                           output_len=sequence_size,
                                           multi_targets=multi_targets,
                                           probabilistic=probabilistic,
                                           scaler=scaler)
                    if probabilistic:
                        output, output_std = output[0], output[1]
                        output, output_std = output[:, :, 0], output_std[0]
                        output_dist = torch.distributions.Normal(output, output_std)
            else:
                if probabilistic:
                    output_dist = model(src.float())
                    output = output_dist.mean.detach().numpy()
                    output_std = output_dist.stddev.detach().numpy()
                else:
                    output = model(src.float())
                    mod_output_list.append(output)
            if type(model).__name__ == "Informer":
                output = output[:, :, 0:multi_targets]
            elif classification:
                labels = targ
                label_list.append(labels)
            elif multi_targets == 1:
                labels = targ[:, :, 0]
            elif multi_targets > 1:
                labels = targ[:, :, 0:multi_targets]
            validation_dataset = validation_loader.dataset
            for crit in criterion:
                if validation_dataset.scale:
                    # Should this also do loss.item() stuff?
                    if len(src.shape) == 2:
                        src = src.unsqueeze(0)
                    src1 = src[:, :, 0:multi_targets]
                    loss_unscaled_full = compute_loss(labels, output, src1, crit, validation_dataset,
                                                      probabilistic, output_std, m=multi_targets)
                    unscaled_crit[crit] += loss_unscaled_full.item() * len(labels.float())
                loss = compute_loss(labels, output, src, crit, False, probabilistic, output_std, m=multi_targets)
                scaled_crit[crit] += loss.item() * len(labels.float())
    if use_wandb:
        if loss_unscaled_full:
            scaled = {k.__class__.__name__: v / (len(validation_loader.dataset) - 1) for k, v in scaled_crit.items()}
            newD = {k.__class__.__name__: v / (len(validation_loader.dataset) - 1) for k, v in unscaled_crit.items()}
            wandb.log({'epoch': epoch,
                       val_or_test: scaled,
                       "unscaled_" + val_or_test: newD})
        else:
            scaled = {k.__class__.__name__: v / (len(validation_loader.dataset) - 1) for k, v in scaled_crit.items()}
            wandb.log({'epoch': epoch, val_or_test: scaled})
    if classification:
        print("Plotting test classification metrics")
        label_list = torch.cat(label_list)
        label_list = label_list[:, 0, :]
        mod_output1 = torch.cat(mod_output_list)[:, 0, :]
        fin = label_list.max(dim=1)[1]
        wandb.log({"roc_" + str(epoch): wandb.plot.roc_curve(fin, mod_output1, classes_to_plot=None, labels=None,
                                                             title="roc_" + str(epoch))})
        wandb.log({"pr": wandb.plot.pr_curve(fin, mod_output1)})
        wandb.log({"conf_": wandb.plot.confusion_matrix(probs=mod_output1.detach().cpu().numpy(),
                   y_true=fin.detach().cpu().numpy(), class_names=None)})
    model.train()
    return list(scaled_crit.values())[0]
