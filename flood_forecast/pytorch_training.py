import torch
import torch.optim as optim
from typing import Type, Dict
from torch.utils.data import DataLoader
import json
import wandb
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.model_dict_function import pytorch_opt_dict, pytorch_criterion_dict
from flood_forecast.transformer_xl.transformer_basic import greedy_decode
from flood_forecast.basic.linear_regression import simple_decode
from flood_forecast.training_utils import EarlyStopper


def train_transformer_style(
        model: PyTorchForecast,
        training_params: Dict,
        takes_target=False,
        forward_params: Dict = {},
        model_filepath: str = "model_save") -> None:
    """
    Function to train any PyTorchForecast model
    :model The initialized PyTorchForecastModel
    :training_params_dict A dictionary of the parameters needed to train model
    :takes_target boolean: Determines whether to pass target during training
    :forward_params: A dictionary for additional forward parameters (for instance target)
    """
    use_wandb = model.wandb
    es = None
    if "early_stopping" in model.params:
        es = EarlyStopper(model.params["early_stopping"]['patience'])
    opt = pytorch_opt_dict[training_params["optimizer"]](
        model.model.parameters(), **training_params["optim_params"])
    criterion_init_params = {}
    if "criterion_params" in training_params:
        criterion_init_params = training_params["criterion_params"]
    criterion = pytorch_criterion_dict[training_params["criterion"]](**criterion_init_params)
    max_epochs = training_params["epochs"]
    data_loader = DataLoader(
        model.training,
        batch_size=training_params["batch_size"],
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None)
    validation_data_loader = DataLoader(
        model.validation,
        batch_size=training_params["batch_size"],
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None)
    test_data_loader = DataLoader(model.test_data, batch_size=1, shuffle=False, sampler=None,
                                  batch_sampler=None, num_workers=0, collate_fn=None,
                                  pin_memory=False, drop_last=False, timeout=0,
                                  worker_init_fn=None)
    meta_model = None
    meta_representation = None
    if "meta_data" in model.params:
        with open(model.params["meta_data"]["path"]) as f:
            json_data = json.load(f)
        dataset_params2 = json_data["dataset_params"]
        training_path = dataset_params2["training_path"]
        valid_path = dataset_params2["validation_path"]
        name = json_data["model_name"]
        meta_model = PyTorchForecast(name, training_path, valid_path, dataset_params2["test_path"], json_data)
        meta_representation = get_meta_representation(model.params["meta_data"]["column_id"],
                                                      model.params["meta_data"]["uuid"], meta_model)
    if use_wandb:
        wandb.watch(model.model)
    session_params = []
    for epoch in range(max_epochs):
        total_loss = torch_single_train(
            model,
            opt,
            criterion,
            data_loader,
            takes_target,
            meta_model,
            meta_representation,
            forward_params)
        print("The loss for epoch " + str(epoch))
        print(total_loss)
        use_decoder = False
        if "use_decoder" in model.params:
            use_decoder = True
        valid = compute_validation(
            validation_data_loader,
            model.model,
            epoch,
            model.params["dataset_params"]["forecast_length"],
            criterion,
            model.device,
            meta_model=meta_model,
            decoder_structure=use_decoder,
            use_wandb=use_wandb)
        if valid < 0.01:
            raise("Error validation loss is zero there is a problem with the validator.")
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
    if model.params["dataset_params"]["class"] != "default":
        decoder_structure = False
    test = compute_validation(
        test_data_loader,
        model.model,
        epoch,
        model.params["dataset_params"]["forecast_length"],
        criterion,
        model.device,
        meta_model=meta_model,
        decoder_structure=decoder_structure,
        use_wandb=use_wandb,
        val_or_test="test_loss")
    print("test loss:", test)
    model.params["run"] = session_params
    model.save_model(model_filepath, max_epochs)


def get_meta_representation(column_id: str, uuid: str, meta_model):
    return meta_model.test_data.__getitem__(0, uuid, column_id)[0]


def torch_single_train(model: PyTorchForecast,
                       opt: optim.Optimizer,
                       criterion: Type[torch.nn.modules.loss._Loss],
                       data_loader: DataLoader,
                       takes_target: bool,
                       meta_data_model: PyTorchForecast,
                       meta_data_model_representation: torch.Tensor,
                       forward_params: Dict = {}) -> float:
    i = 0
    running_loss = 0.0
    for src, trg in data_loader:
        opt.zero_grad()
        # Convert to CPU/GPU/TPU
        src = src.to(model.device)
        trg = trg.to(model.device)
        # TODO figure how to avoid
        if meta_data_model:
            representation = meta_data_model.model.generate_representation(meta_data_model_representation)
            forward_params["meta_data"] = representation
        if takes_target:
            forward_params["t"] = trg
        output = model.model(src, **forward_params)
        labels = trg[:, :, 0]
        loss = criterion(output, labels.float())
        if loss > 100:
            print("Warning: high loss detected")
        loss.backward()
        opt.step()
        if torch.isnan(loss) or loss == float('inf'):
            raise("Error infinite or NaN loss detected. Try normalizing data or performing interpolation")
        running_loss += loss.item()
        i += 1
    print("The running loss is:")
    print(running_loss)
    print("The number of items in train is: ")
    print(i)
    total_loss = running_loss / float(i)
    return total_loss


def compute_validation(validation_loader: DataLoader,  # s lint
                       model,
                       epoch: int,
                       sequence_size: int,
                       criterion: Type[torch.nn.modules.loss._Loss],
                       device: torch.device,
                       decoder_structure=False,
                       meta_data_model=None,
                       use_wandb: bool = False,
                       meta_model=None,
                       val_or_test="validation_loss") -> float:
    """
    Function to compute the validation or test loss
    """
    model.eval()
    loop_loss = 0.0
    with torch.no_grad():
        i = 0
        loss_unscaled_full = 0.0
        for src, targ in validation_loader:
            src = src.to(device)
            targ = targ.to(device)
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
                else:
                    output = simple_decode(model, src, targ.shape[1], targ, 1)[:, :, 0]
            else:
                output = model(src.float())
            labels = targ[:, :, 0]
            validation_dataset = validation_loader.dataset
            if validation_dataset.scale:
                # unscaled_src = validation_dataset.scale.inverse_transform(src.cpu())
                unscaled_out = validation_dataset.inverse_scale(output.cpu())
                unscaled_labels = validation_dataset.inverse_scale(labels.cpu())
                loss_unscaled = criterion(unscaled_out, unscaled_labels.float())
                loss_unscaled_full += len(labels.float()) * loss_unscaled.item()
                if i % 10 == 0 and use_wandb:
                    import wandb
                    wandb.log({"trg": unscaled_labels, "model_pred": unscaled_out})
            loss = criterion(output, labels.float())
            loop_loss += len(labels.float()) * loss.item()
    if use_wandb:
        import wandb
        if loss_unscaled_full:
            tot_unscaled_loss = loss_unscaled_full / (len(validation_loader.dataset) - 1)
            wandb.log({'epoch': epoch,
                       val_or_test: loop_loss / (len(validation_loader.dataset) - 1),
                       "unscaled_" + val_or_test: tot_unscaled_loss})
        else:
            wandb.log({'epoch': epoch, val_or_test: loop_loss /
                       (len(validation_loader.dataset) - 1)})
    model.train()
    return loop_loss / (len(validation_loader.dataset) - 1)
