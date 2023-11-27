## Taken from Karpathy's github: [url]

print('Importing modules...')#, end=' ')
import argparse
import copy
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
print('-> Done.')

print('Importing local files...')#, end=' ')
sys.path.append('../../../src/')
# sys.path.append(os.path.join('..', '..', '..', 'src', 'model'))
from model.model import Model
from continuous_model.continuous_model import ContinuousModel
from src_utils.filter_dict import filter_keys

from argument_parsing import parse_arguments, assert_and_correct_arguments
import data
from utils import obtain_model_name, load_model, generate_text
print('-> Done.')

# torch.set_default_dtype(torch.float64)

print('Parsing arguments...')#, end=' ')
args = parse_arguments()
assert_and_correct_arguments(args)
print('-> Done.')
print(f'args: {args}')

_vars = copy.deepcopy(args)

## This here below must change
sys.path.append(f'model_architectures/{_vars.model_name}/methods/')
from generate import generate
from init_weights import init_weights

DATA_DIR = os.path.join('..', 'data')

def main():
  # hyperparameters
  training_batches = 1000#500
  evaluation_batches = 200
  learning_rate = float(_vars.learning_rate)
  _vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'device {_vars.device}')
  _vars.dropout = 0.#0.2
  # ------------

  if _vars.debug:
    _vars.batch_size = 2
    _vars.context_window = 5
    training_batches = 3
    evaluation_batches = 1
    _vars.max_new_tokens = 10
    _vars.model_dimension = 8#32
    _vars.num_heads = 4
    # _vars.continuous = True

  # print(_vars)

  torch.manual_seed(_vars.seed)

  ## DATA
  print('\n1. Loading data')
  data.obtain_data(DATA_DIR, _vars)

  ## MODEL
  print('\n2. Building model')
  print(f'Building model w/ {_vars.num_layers} decoder layers')
  continuous_blocks_num_layers = [_vars.num_layers]
  _vars.model = Model(
    continuous_blocks_num_layers=continuous_blocks_num_layers, 
    initialize_weights=False, **_vars.__dict__,
  )

  # torch.manual_seed(_vars.seed)
  # model.apply(init_weights)

  if _vars.continuous:
    print(' 2.1 Building continuous model')
    _vars.model = ContinuousModel(**_vars.__dict__)

  # print(model)
  # print the number of parameters in the model
  print(sum(p.numel() for p in _vars.model.parameters())/1e6, 'M parameters')

  ## Debug: compare model weights
  # for p in m.parameters():
  #   print(p.shape, p.ravel()[:10])
  # sys.exit()

  # create a PyTorch optimizer
  _vars.optimizer = torch.optim.AdamW(
    _vars.model.parameters(), lr=learning_rate,
  )
  _vars.criterion = nn.CrossEntropyLoss()

  model_name1, model_name2 = obtain_model_name(_vars)
  if _vars.load: load_model(
    _vars.model, _vars.optimizer, model_name1, model_name2,
  )

  # if k != 0:
  #   print(f'Interpolating weights from previous model to the new one')
  #   interpolate_weights(model, old_model)

  torch.manual_seed(_vars.seed)

  print(f'\n3. Training model w/ {_vars.num_layers} decoder layers')

  def get_batch(split):
    data = _vars.data_sets[split]
    ix = torch.randint(len(data) - _vars.context_window, (_vars.batch_size,))
    x = torch.stack([data[i   : i   + _vars.context_window] for i in ix])
    y = torch.stack([data[i+1 : i+1 + _vars.context_window] for i in ix])
    x, y = x.to(_vars.device), y.to(_vars.device)
    return x, y

  num_epochs_list = [int(num_epochs) \
                     for num_epochs in _vars.num_epochs.split('-')]

  for num_epochs in num_epochs_list:
    for epoch in range(num_epochs + 1):
      # torch.manual_seed(batch_idx)

      ## Train
      if epoch > 0:
        training_output = _vars.model.train_(
          num_batches=training_batches, compute_accuracy=False, 
          print_times=False, get_batch=lambda: get_batch('training'), 
          **filter_keys(_vars.__dict__, ('model',)),
        )

      ## Evaluate
      evaluation_output = _vars.model.evaluate(
        num_batches=evaluation_batches, compute_accuracy=False, 
        print_times=False, get_batch=lambda: get_batch('validation'), 
        **filter_keys(_vars.__dict__, ('model',)),
      )

      if epoch > 0: print(epoch, training_output, evaluation_output)
      else        : print(epoch,                  evaluation_output)

  if _vars.generate:
    print('\n4. Generating text')
    # generate from the model
    generate_text(**_vars.__dict__)

if __name__ == '__main__':
  main()




