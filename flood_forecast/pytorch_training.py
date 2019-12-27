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

def train_transformer_style(model: PyTorchForecast, training_params: Dict, forward_params = {}):
  """
  Function to train any PyTorchForecast model  
  :model The initialized PyTorchForecastModel
  :training_params_dict
  """
  use_wandb = model.wandb
  opt = pytorch_opt_dict[training_params["optimizer"]](model.model.parameters())
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
  #criterion = torch.nn.MSELoss()
  #optimizer = torch.optim.Adam(a.parameters())
  if use_wandb:
    import wandb
    wandb.watch(model.model)
  session_params = []
  for epoch in range(max_epochs):
      i = 0
      running_loss = 0.0
      for src, trg in data_loader:
          opt.zero_grad()
          output = model.model(src, **forward_params)
          labels = trg[:, :, 0] 
          loss = criterion(output, labels.float())
          if loss > 10:
              print(src)
          loss.backward()
          #torch.nn.utils.clip_grad_norm_(s.parameters(), 0.5)
          opt.step()
          running_loss += loss.item()
          i+=1
          if torch.isnan(loss) or loss==float('inf'):
              raise "Error infinite or NaN loss detected. Try normalizing data or performing interpolation"
      print("The loss is")
      print(loss)
      use_decoder = False
      if "use_decoder" in  model.params["model_params"]:
        use_decoder = True
      valid = compute_validation(validation_data_loader, model.model, epoch, model.params["dataset_params"]["forecast_length"], criterion, decoder_structure=use_decoder, use_wandb=use_wandb)
      if use_wandb:
        wandb.log({'epoch': epoch, 'loss': loss/i})
      epoch_params = {"epoch":epoch, "train_loss":str(loss/i), "validation_loss":str(valid)} 
      session_params.append(epoch_params)
  model.params["run"] = session_params
  model.save_model("model_save")


def compute_validation(validation_loader, model, epoch, sequence_size, criterion, decoder_structure=False, use_wandb=False):
  model.eval()
  mask = generate_square_subsequent_mask(sequence_size)
  loop_loss = 0.0
  with torch.no_grad():
    i = 0
    for src, targ in validation_loader:
      i+=1
      if decoder_structure:
        src_mask = generate_square_subsequent_mask(sequence_size)
        output = greedy_decode(model, src, src_mask, sequence_size, targ, src)[:, :, 0]

      # To do implement greedy decoding
      # https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
    else:
      output = model(src.float(), mask)
      labels = targ[:, :, 0]
    loss = criterion(output, labels.float())
    loop_loss += len(labels.float())*loss.item()
  if use_wandb:
    import wandb
    wandb.log({'epoch': epoch, 'validation_loss': loop_loss/(len(validation_loader.dataset)-1)})
  model.train()
  return loop_loss/(len(validation_loader.dataset)-1)