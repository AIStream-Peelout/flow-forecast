import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from typing import Type, Dict
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.model_dict_function import pytorch_opt_dict, pytorch_criterion_dict
from flood_forecast.model_dict_function import generate_square_subsequent_mask
from flood_forecast.transformer_xl.transformer_basic import greedy_decode
from flood_forecast.basic.linear_regression import simple_decode

def train_transformer_style(model: PyTorchForecast, training_params: Dict, takes_target=False, forward_params:Dict = {})->None:
  """
  Function to train any PyTorchForecast model  
  :model The initialized PyTorchForecastModel
  :training_params_dict A dictionary of the parameters needed to train model
  :takes_target boolean: Determines whether to pass target during training
  :forward_params: A dictionary for additional forward parameters (for instance target)
  """
  use_wandb = model.wandb
  opt = pytorch_opt_dict[training_params["optimizer"]](model.model.parameters(), **training_params["optim_params"])
  criterion = pytorch_criterion_dict[training_params["criterion"]]
  max_epochs = training_params["epochs"]
  data_loader = DataLoader(model.training, batch_size=training_params["batch_size"], shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
  validation_data_loader = DataLoader(model.validation, batch_size=training_params["batch_size"], shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
  if use_wandb:
    import wandb
    wandb.watch(model.model)
  session_params = []
  model_save_path_arr = []
  prev_valid_loss = 100000000
  for epoch in range(max_epochs):
      total_loss = torch_single_train(model, opt, criterion, data_loader, takes_target, forward_params)
      print("The loss for epoch " + str(epoch))
      print(total_loss)
      use_decoder = False
      if "use_decoder" in model.params:
        use_decoder = True
      valid = compute_validation(validation_data_loader, model.model, epoch, model.params["dataset_params"]["forecast_length"], criterion, model.device, decoder_structure=use_decoder, use_wandb=use_wandb)
      if valid <0.01:
        raise "Error validation loss is zero there is a problem with the validator."
      if use_wandb:
        wandb.log({'epoch': epoch, 'loss': total_loss})
      if "early_stopping" in model.params:
        stopping_params = model.params["early_stopping"]
        stop_now = True
        patience = model.params["early_stopping"]["patience"]
        if len(session_params) > patience:
          for i in range(0, int(patience)):
            if session_params[len(session_params)-i]["improved"]:
              stop_now = False
          else: 
            stop_now = False
        best_val_loss = sorted(session_params, key = lambda x: x["validation_loss"])
        if stop_now:
          # TODO load model with best validation loss
          epoch = best_val_loss[0]["epoch"]

          model.load_model(model.params["model_base"], model.params, weight_path=model_save_path_arr[epoch])
          print("Stopping due to no improvement")
          print("weights epoch: " + int(epoch))
          break
        model_save = model.save_model("model_epoch", epoch)
        model_save_path_arr.append(model_save)

      epoch_params = {"epoch":epoch, "train_loss":str(total_loss), "validation_loss":str(valid), "improved":valid<prev_valid_loss} 
      session_params.append(epoch_params)
      prev_valid_loss = valid
  model.params["run"] = session_params
  model.save_model("model_save", max_epochs)

def torch_single_train(model:PyTorchForecast, opt:optim.Optimizer, criterion:Type[torch.nn.modules.loss._Loss], data_loader:DataLoader, takes_target:bool, forward_params:Dict={})->float:
  i = 0
  running_loss = 0.0
  for src, trg in data_loader:
    opt.zero_grad()
    # Convert to CPU/GPU/TPU 
    src = src.to(model.device)
    trg = trg.to(model.device)
    # TODO figure how to avoid
    if takes_target:
      forward_params["t"] = trg 
    output = model.model(src, **forward_params)
    labels = trg[:, :, 0] 
    loss = criterion(output, labels.float())
    if loss > 100:
      print("Warning: high loss detected")
    loss.backward()
    opt.step()
    if torch.isnan(loss) or loss==float('inf'):
        raise "Error infinite or NaN loss detected. Try normalizing data or performing interpolation"
    running_loss += loss.item()
    i+=1
  print("The running loss is:")
  print(running_loss)
  print("The number of items in train is: ")
  print(i)
  total_loss = running_loss/float(i)
  return total_loss

def compute_validation(validation_loader:DataLoader, model, epoch:int, sequence_size:int, criterion:Type[torch.nn.modules.loss._Loss], device:torch.device, decoder_structure=False, use_wandb:bool=False)->float:
  model.eval()
  loop_loss = 0.0
  with torch.no_grad():
    i = 0
    loss_unscaled_full = 0.0
    for src, targ in validation_loader:
      src = src.to(device)
      targ = targ.to(device)
      i+=1
      if decoder_structure:
        if hasattr(model, "mask"):
          targ_clone = targ.detach().clone()
          output = greedy_decode(model, src, sequence_size, targ_clone, device=device)[:, :, 0]
        else:
          output = simple_decode(model, src, sequence_size, targ, 1)[:, :, 0]
      else:
        output = model(src.float())
      labels = targ[:, :, 0]
      validation_dataset = validation_loader.dataset
      if validation_dataset.scale:
        #unscaled_src = validation_dataset.scale.inverse_transform(src.cpu())
        unscaled_out = validation_dataset.inverse_scale(output.cpu())
        unscaled_labels = validation_dataset.inverse_scale(labels.cpu())
        loss_unscaled = criterion(unscaled_out, unscaled_labels.float())
        loss_unscaled_full += len(labels.float())*loss_unscaled.item()
        if i%10 ==0 and use_wandb:
          import wandb
          wandb.log({"trg":unscaled_labels, "model_pred":unscaled_out})
      loss = criterion(output, labels.float())
      loop_loss += len(labels.float())*loss.item()
  if use_wandb:
    import wandb
    if loss_unscaled_full:
      tot_unscaled_loss = loss_unscaled_full/(len(validation_loader.dataset)-1)
      wandb.log({'epoch': epoch, 'validation_loss': loop_loss/(len(validation_loader.dataset)-1), "unscaled_loss": tot_unscaled_loss})
    else:
      wandb.log({'epoch': epoch, 'validation_loss': loop_loss/(len(validation_loader.dataset)-1)})
  model.train()
  return loop_loss/(len(validation_loader.dataset)-1)