## Taken from Karpathy's github: [url]

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

sys.path.append('../../../src/')

from argument_parsing import parse_arguments
from continuous_model.continuous_model import ContinuousModel
import data
from model.model import Model
from utils import estimate_loss, get_batch

args = parse_arguments()

## This here below must change
sys.path.append(f'model_architectures/{args.model_name}/methods/')
from generate import generate
from init_weights import init_weights

def main():
  # hyperparameters
  max_iters = int(args.num_epochs)
  eval_interval = 500
  learning_rate = float(args.lr)
  max_new_tokens = 500
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(device)
  eval_iters = 200
  n_embd = args.model_dimension
  n_head = args.num_heads
  n_layer = args.N
  T = args.T if args.T is not None else n_layer
  dropout = 0.2
  seed = args.seed
  # ------------

  if args.debug:
    max_iters = 10#100
    args.batch_size = 2
    block_size = 8
    eval_interval = 1
    eval_iters = 1
    max_new_tokens = 10
    n_embd = 32
    n_head = 4

  print(args)

  torch.manual_seed(seed)

  ## DATA
  print('\n1. Loading data')
  train_data, val_data, decode, vocab_size = data.main()

  ## MODEL
  print('\n2. Building model')
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
    block_size=args.max_len,
  )
  torch.manual_seed(seed)
  model.apply(init_weights)
  
  if args.continuous:
    print(' 2.1 Building continuous model')
    model = ContinuousModel(
      model=model, T=T, solver=args.solver,
    )

  m = model.to(device)
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

  ## Debug: compare model weights
  # for p in m.parameters():
  #   print(p.shape, p.ravel()[-10:])
  # sys.exit()

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()

  # if k != 0:
  #   print(f'Interpolating weights from previous model to the new one')
  #   interpolate_weights(model, old_model)
    
  torch.manual_seed(seed)

  print(f'\n3. Training model w/ {n_layer} decoder layers')
  for epoch in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if epoch % eval_interval == 0 or epoch == max_iters - 1:
        losses = estimate_loss(
          model, eval_iters, train_data, val_data, args.max_len, 
          args.batch_size, device, criterion
        )
        print(
          f"Epoch {epoch}: train loss {losses['train']:.4f}, " \
        + f"val loss {losses['val']:.4f}"
        )

    batch = get_batch('train', train_data, val_data, args.max_len, args.batch_size, device)
    input_ids, target_ids = batch
    model_inputs = {'input': input_ids, 'target': target_ids,
                    'criterion': criterion, 'compute_accuracy': False}
    model_outputs = model(**model_inputs)
    loss = model_outputs['loss']

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  print('\n4. Generating text')
  # generate from the model
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  print(
    decode(
      generate(
        model=m, x=context, max_new_tokens=max_new_tokens, 
        block_size=args.max_len,
      )[0].tolist()
    )
  )
  #open('../data/more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

if __name__ == '__main__':
  main()


