## Taken from Karpathy's github: [url]

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

sys.path.append('../../../src/')

from continuous_model.continuous_model import ContinuousModel
import data
from model.model import Model
from utils import estimate_loss, get_batch

parser = argparse.ArgumentParser()
parser.add_argument('--continuous', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--model_name', type=str, default='transformer')
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--num_epochs', type=int, default=5000)
parser.add_argument('--solver', type=str, default='Forward Euler')
parser.add_argument('--T', type=float, default=None)
args = parser.parse_args()

## This here below must change
sys.path.append(f'model_architectures/{args.model_name}/methods/')
from generate import generate

def main():
  # hyperparameters
  batch_size = 64 # how many independent sequences will we process in parallel?
  block_size = 256 # what is the maximum context length for predictions?
  max_iters = args.num_epochs
  eval_interval = 500
  learning_rate = 3e-4
  max_new_tokens = 500
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(device)
  eval_iters = 200
  n_embd = 384
  n_head = 6
  n_layer = args.num_layers
  T = args.T if args.T is not None else n_layer
  dropout = 0.2
  seed = 1337
  # ------------

  if args.debug:
    max_iters = 10#100
    batch_size = 2
    block_size = 8
    eval_interval = 1
    eval_iters = 1
    max_new_tokens = 10
    n_embd = 32
    n_head = 4

  torch.manual_seed(seed)

  ## DATA
  train_data, val_data, decode, vocab_size = data.main()

  ## MODEL
  print(f'Building model w/ {n_layer} decoder layers')
  model_architecture_path = '.'.join(
  ['model_architectures', args.model_name, 'architecture']
  )
  model = Model(
    model_architecture_path=model_architecture_path, 
    N=n_layer,
    d_model=n_embd,
    n_head=n_head,
    dropout=dropout,
    vocab_size=vocab_size,
    block_size=block_size,
  )
  if args.continuous:
    model = ContinuousModel(
      model=model, T=T, solver=args.solver,
    )
  m = model.to(device)
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  # if k != 0:
  #   print(f'Interpolating weights from previous model to the new one')
  #   interpolate_weights(model, old_model)
    
  torch.manual_seed(seed)

  print(f'Training model w/ {n_layer} decoder layers')
  for epoch in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if epoch % eval_interval == 0 or epoch == max_iters - 1:
        losses = estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device)
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device)

    # evaluate the loss
    model_inputs = {'x': xb, 'targets': yb}
    outputs = model(**model_inputs)#xb, yb)
    logits, loss = outputs['x'], outputs['loss']
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  # generate from the model
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  print(
    decode(
      generate(
        model=m, x=context, max_new_tokens=max_new_tokens, block_size=block_size
      )[0].tolist()
    )
  )
  #open('../data/more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

if __name__ == '__main__':
  main()


