
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from typing import Type
from torch.nn.modules.loss import _Loss

def train_transformer_style(model: Type[torch.nn.module], max_expochs: int, criterion: Type[torch.nn._Loss]):
  """
  Function to train a PyTorch model 
  """
  #criterion = torch.nn.MSELoss()
  #optimizer = torch.optim.Adam(a.parameters())
  if use_wandb:
    wandb.watch(model)
  for epoch in range(max_epochs):
      i = 0
      running_loss = 0.0
      for src, trg in data_loader:
          mask = generate_square_subsequent_mask(sequence_size)
          optimizer.zero_grad()
          #print(src)
          output = a(src.float(), mask)
          #output = s(src.float(), trg.float(), mask)
          labels = trg[:, :, 0] 
          loss = criterion(output.view(-1, sequence_size), labels.float())
          if loss > 10:
              print(src)
          #print(loss)
          loss.backward()
          #torch.nn.utils.clip_grad_norm_(s.parameters(), 0.5)
          optimizer.step()
          running_loss += loss.item()
          i+=1
          if torch.isnan(loss) or loss==float('inf'):
              print(i)
              break
      print("The loss is")
      print(loss)
      print(compute_validation(validation_data_loader, a, epoch, sequence_size, criterion))
      wandb.log({'epoch': epoch, 'loss': loss/i})

def compute_trans_validation(validation_loader, model, epoch, sequence_size, criterion, decoder_structure=False):
    model.eval()
    mask = generate_square_subsequent_mask(sequence_size)
    loop_loss = 0.0
    print(loop_loss)
    with torch.no_grad():
        i = 0 
        for src, targ in validation_loader:
            i+=1
            if decoder_structure:
                output = model(src.float(), trg.float(), mask)
                # To do implement greedy decoding
                # https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
            else: 
                output = model(src.float(), mask)
            labels = targ[:, :, 0]
            loss = criterion(output.view(-1, sequence_size), labels.float())
            loop_loss += len(labels.float())*loss.item()
        wandb.log({'epoch': epoch, 'validation_loss': loop_loss/(len(validation_data_loader.dataset)-1)})
    model.train()
    return loop_loss/(len(validation_data_loader.dataset)-1)
