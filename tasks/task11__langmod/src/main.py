## Taken from Karpathy's github: [url]

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

sys.path.append('../../../src/')

from argument_parsing import parse_arguments, assert_and_correct_arguments
from continuous_model.continuous_model import ContinuousModel
import data
from model.model import Model
from utils import estimate_loss, get_batch, train_batch

# torch.set_default_dtype(torch.float64)

args = parse_arguments()
assert_and_correct_arguments(args)

## This here below must change
sys.path.append(f'model_architectures/{args.model_name}/methods/')
from generate import generate
from init_weights import init_weights

DATA_DIR = os.path.join('..', 'data')

def obtain_model_name(args):
  # model_name = '_'.join([
  #   f'{k}{args.__dict__[k]}'.replace(' ', '_') \
  #   for k in sorted(args.__dict__.keys())
  # ]) # str(args)
  model_name = ''
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
    if v == 'wikipedia': v = 'wiki'
    if k == 'levels_scheme': k = 'scheme'
    if k == 'save': continue
    if k == 'model_dimension': k = 'd'
    if k == 'model_name': k = ''
    if k == 'num_epochs': k = 'epochs'
    if k == 'num_heads': k = 'H'
    if v == 'Forward Euler': v = 'FE'
    if k == 'tokenization': k = 'tok'
    if v == 'character': v = 'char'
    if k == 'load': continue
    model_name += f'_{k}{v}'
  model_name = model_name[1:]
  model_name1 = model_name + '_copy1'
  model_name2 = model_name + '_copy2'
  return model_name1, model_name2

def load_model(model, optimizer, model_name1, model_name2):
  other_states = {}
  try:
    print('Loading model, copy1')
    other_states = model.load(model_name=model_name1, optimizer=optimizer)
    print('other_states', other_states)
  except: 
    try:
      print('Loading model, copy2')
      other_states = model.load(model_name=model_name2, optimizer=optimizer)
    except:
      # print('The model could not be loaded because of an unknown error.')
      other_states['error'] = 'Unknown error.'
  if 'error' in other_states: print(f"Error: {other_states['error']}")
  else: print('Model successfully loaded.')


def generate_text(m, device, decode, max_new_tokens, **kwargs):
  m.eval()
  bos_token = '<|endoftext|>'
  bos_token_id = 50256#tokenizer('<|endoftext|>')['input_ids'][0]
  context = torch.empty((1, 1), dtype=torch.long, device=device).fill_(bos_token_id)
  print(
    decode(
      generate(
        model=m, x=context, max_new_tokens=max_new_tokens, **kwargs,
      )[0].tolist()
    )
  )
  #open('../data/more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

def main():
  # hyperparameters
  num_batch_passes = int(args.num_epochs)
  eval_interval = 1000#500
  learning_rate = float(args.lr)
  max_new_tokens = 500
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'device {device}')
  eval_iters = 200
  T = args.T
  dropout = 0.#0.2
  seed = args.seed
  # ------------

  if args.debug:
    num_batch_passes = 2#1#10#100
    args.batch_size = 2
    args.context_window = 5
    eval_interval = 1
    eval_iters = 1
    max_new_tokens = 10
    args.model_dimension = 32
    args.num_heads = 4
    # args.continuous = True


  print(args)

  _vars = args.__dict__.copy()

  torch.manual_seed(seed)

  ## DATA
  print('\n1. Loading data')
  # train_data, val_data, decode, vocabulary_size = \
  #   data.tokenize_data_at_character_level(DATA_PATH)
  train_data, val_data, decode, vocabulary_size = \
    data.obtain_data(DATA_DIR, args.input_text, args.tokenization, args.debug)

  ## MODEL
  print('\n2. Building model')
  print(f'Building model w/ {args.num_layers} decoder layers')
  model_architecture_path = '.'.join(
    ['model_architectures', args.model_name, 'architecture']
  )
  model = Model(
    model_architecture_path=model_architecture_path, 
    num_layers=args.num_layers,
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
    model = ContinuousModel(model=model, T=T, solver=args.ode_solver)
    
  # print(model)

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

  model_name1, model_name2 = obtain_model_name(args)

  if args.load: load_model(model, optimizer, model_name1, model_name2)

  # if k != 0:
  #   print(f'Interpolating weights from previous model to the new one')
  #   interpolate_weights(model, old_model)
    
  torch.manual_seed(seed)

  print(f'\n3. Training model w/ {args.num_layers} decoder layers')
  model.train()
  model_save_t0 = time.time()
  train_bf_eval_t0 = time.time()
  for batch_idx in range(num_batch_passes+1):
    # batch_start_time = time.time()
    # print(f'batch_idx {batch_idx}')

    torch.manual_seed(batch_idx)

    if batch_idx > 0:
      t0_batch = time.time()
      train_batch(
        model, train_data, val_data, device, optimizer, criterion, 
        **args.__dict__,
      )
      print(f'Batch time: {time.time() - t0_batch}')

      # batch_end_time = time.time()
      # print(f'Batch training time: {batch_end_time - batch_start_time :.2f} seconds')

      ## Save & generate some text every 5 minutes
      if args.save and time.time() - model_save_t0 > 5*60:
        t1 = time.time()
        model.save(
          fn_without_extension=model_name1, optimizer=optimizer,
        )
        model.save(
          fn_without_extension=model_name2, optimizer=optimizer,
        )
        print(f'Models-saving time: {time.time() - t1}')

        model_save_t0 = time.time()

    # every once in a while evaluate the loss on train and val sets
    if batch_idx % eval_interval == 0 or batch_idx == num_batch_passes:
      evaluation_t0 = time.time()
      if batch_idx != 0:
        print(
          f'Training time until evaluation: ' + \
          f'{evaluation_t0 - train_bf_eval_t0}'
        )

      losses = estimate_loss(
        model, eval_iters, train_data, val_data, device, criterion, 
        **args.__dict__,
      )
      print(
        f"Batch {batch_idx}: train loss {losses['train'] :.8f}, " \
      + f"val loss {losses['val'] :.8f}"
      )

      print(f'Evaluation time: {time.time() - evaluation_t0}')

      # generation_t0 = time.time()
      # print('Generate some text:')
      # generate_text(m, device, decode, max_new_tokens, **args.__dict__)
      # print(f'Text-generation time: {time.time() - generation_t0}')
      train_bf_eval_t0 = time.time()

  if args.generate:
    print('\n4. Generating text')
    # generate from the model
    generate_text(m, device, decode, max_new_tokens, **args.__dict__)

if __name__ == '__main__':
  main()


