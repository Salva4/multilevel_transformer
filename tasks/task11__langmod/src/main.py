## Taken from Karpathy's github: [url]

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

sys.path.append('../../../src/')

from argument_parsing import parse_arguments
from continuous_model.continuous_model import ContinuousModel
import data
from model.model import Model
from utils import estimate_loss, get_batch, train_batch

args = parse_arguments()

## This here below must change
sys.path.append(f'model_architectures/{args.model_name}/methods/')
from generate import generate
from init_weights import init_weights

DATA_DIR = os.path.join('..', 'data')

def main():
  # hyperparameters
  num_batch_passes = int(args.num_epochs)
  eval_interval = 500
  learning_rate = float(args.lr)
  max_new_tokens = 500
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(device)
  eval_iters = 200
  T = args.T if args.T is not None else args.N
  dropout = 0.#0.2
  seed = args.seed
  # ------------

  if args.debug:
    num_batch_passes = 10#100
    args.batch_size = 2
    context_window = 8
    eval_interval = 1
    eval_iters = 1
    max_new_tokens = 10
    args.model_dimension = 32
    args.num_heads = 4

  print(args)

  torch.manual_seed(seed)

  ## DATA
  print('\n1. Loading data')
  # train_data, val_data, decode, vocabulary_size = \
  #   data.tokenize_data_at_character_level(DATA_PATH)
  train_data, val_data, decode, vocabulary_size = \
    data.obtain_data(DATA_DIR, args.input_text, args.tokenization)

  ## MODEL
  print('\n2. Building model')
  print(f'Building model w/ {args.N} decoder layers')
  model_architecture_path = '.'.join(
    ['model_architectures', args.model_name, 'architecture']
  )
  model = Model(
    model_architecture_path=model_architecture_path, 
    N=args.N,
    model_dimension=args.model_dimension,
    num_heads=args.num_heads,
    dropout=dropout,
    vocabulary_size=vocabulary_size,
    context_window=args.context_window,
  )
  torch.manual_seed(seed)
  # model.apply(init_weights)

  if args.continuous:
    print(' 2.1 Building continuous model')
    model = ContinuousModel(model=model, T=T, solver=args.solver)

  m = model.to(device)
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

  ## Debug: compare model weights
  # for p in m.parameters():
  #   print(p.shape, p.ravel()[:10])
  # sys.exit()

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()

  # if k != 0:
  #   print(f'Interpolating weights from previous model to the new one')
  #   interpolate_weights(model, old_model)
    
  torch.manual_seed(seed)

  print(f'\n3. Training model w/ {args.N} decoder layers')
  model.train()
  for batch_idx in range(num_batch_passes+1):
    # batch_start_time = time.time()

    torch.manual_seed(batch_idx)

    if batch_idx > 0:
      train_batch(
        model, train_data, val_data, args.context_window, args.batch_size, 
        device, optimizer, criterion,
      )

    # batch_end_time = time.time()
    # print(f'Batch training time: {batch_end_time - batch_start_time :.2f} seconds')

    if args.save:
      # fn_without_extension = '_'.join([
      #   f'{k}{args.__dict__[k]}'.replace(' ', '_') \
      #   for k in sorted(args.__dict__.keys())
      # ]) # str(args)
      fn_without_extension = ''
      for (k, v) in sorted(args.__dict__.items()):
        if v is None: continue
        if k == 'batch_size': k = 'bs'
        if k == 'coarsening_factor': k = 'cf'
        if k == 'context_window': k = 'L'
        if k == 'continuous': k = 'cont'
        if v == False: v = 'F'
        if v == True: v = 'T'
        if k == 'input_text': k = 'text'
        if v == 'shakespeare': v = 'shak'
        if v == 'wikipedia': k = 'wiki'
        if k == 'levels_scheme': k = 'scheme'
        if k == 'save': continue
        if k == 'model_dimension': k = 'd'
        if k == 'model_name': k = ''
        if k == 'num_epochs': k = 'epochs'
        if k == 'num_heads': k = 'H'
        if v == 'Forward Euler': v = 'FE'
        if k == 'tokenization': k = 'tok'
        if v == 'character': v = 'char'
        fn_without_extension += f'_{k}{v}'
      fn_without_extension = fn_without_extension[1:]
      model.save(
        fn_without_extension=fn_without_extension, optimizer=optimizer,
      )

    # every once in a while evaluate the loss on train and val sets
    if batch_idx % eval_interval == 0 or batch_idx == num_batch_passes:
      losses = estimate_loss(
        model, eval_iters, train_data, val_data, args.context_window, 
        args.batch_size, device, criterion
      )
      print(
        f"Batch {batch_idx}: train loss {losses['train'] :.8f}, " \
      + f"val loss {losses['val'] :.8f}"
      )

  if args.generate:
    print('\n4. Generating text')
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(
      decode(
        generate(
          model=m, x=context, max_new_tokens=max_new_tokens, 
          context_window=args.context_window,
        )[0].tolist()
      )
    )
    #open('../data/more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

if __name__ == '__main__':
  main()


