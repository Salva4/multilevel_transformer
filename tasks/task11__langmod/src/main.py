## Adapted from Karpathy's github: https://karpathy.ai/zero-to-hero.html

print('Importing modules...')#, end=' ')
import copy
import torch
import torch.nn as nn
import sys
print('-> Done.\n')

print('Importing local files...')#, end=' ')
sys.path.append('../../../src/')
from model.model import Model
from continuous_model.continuous_model import ContinuousModel
from src_utils.filter_dict import filter_keys
from src_utils.optimizer import initialize_optimizer

from argument_parsing import parse_arguments, assert_and_correct_arguments
from data import obtain_data
from example_utils import obtain_model_name, load_model, generate_text
print('-> Done.\n')

# torch.set_default_dtype(torch.float64)

print('Parsing arguments...')#, end=' ')
args = parse_arguments()
assert_and_correct_arguments(args)
print('-> Done.\n')
print(f'Args: {args}')

_vars = copy.deepcopy(args)

## This here below must change
sys.path.append(f'model_architectures/{_vars.model_name}/methods/')
from generate import generate
from init_weights import init_weights

def main():
  ## debug mode #######################
  if _vars.debug:
    _vars.batch_size = 2
    _vars.context_window = 5
    _vars.max_new_tokens = 10
    _vars.model_dimension = 8#32
    _vars.num_heads = 4
    # _vars.continuous = True
  #####################################

  _vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Device: {_vars.device}\n')
  _vars.dropout = 0.#0.2  <-- careful with MGRIT when dropout > 0.

  torch.manual_seed(_vars.seed)

  ## DATA
  print('1. Loading data')
  obtain_data(_vars)
  print('-> Done.\n')

  ## MODEL
  print('2. Building model')
  continuous_blocks_num_layers = [_vars.num_layers]
  _vars.model = Model(
    continuous_blocks_num_layers=continuous_blocks_num_layers, 
    initialize_weights=False, **_vars.__dict__,
  )
  print('-> Done.\n')

  # model.apply(init_weights)

  if _vars.continuous:
    print(' 2.1 Turning the model continuous')
    continuous_blocks_T = [_vars.T]
    _vars.model = ContinuousModel(
      continuous_blocks_T=continuous_blocks_T, **_vars.__dict__,
    )#.to(device)
    print(' -> Done.\n')

  # print(f'Model: {model}\n')
  # print(
  #   f'Number of model parameters:', 
  #   sum(parameter.numel() for parameter in _vars.model.parameters())/1e6, 
  #   '\n'
  # )

  _vars.optimizer = initialize_optimizer(**_vars.__dict__)
  _vars.criterion = nn.CrossEntropyLoss()

  model_name1, model_name2 = obtain_model_name(_vars)
  if _vars.load:
    load_model(_vars.model, _vars.optimizer, model_name1, model_name2)

  print(f'3. Training model w/ {_vars.num_layers} decoder layers')

  def get_batch(split):
    data = _vars.data_sets[split]
    ix = torch.randint(len(data) - _vars.context_window, (_vars.batch_size,))
    x = torch.stack([data[i   : i   + _vars.context_window] for i in ix])
    y = torch.stack([data[i+1 : i+1 + _vars.context_window] for i in ix])
    x, y = x.to(_vars.device), y.to(_vars.device)

    return x, y

  num_epochs_list    = [  int(num_epochs   ) for num_epochs    in _vars.num_epochs   .split('_')]
  levels_list        = [  int(level        ) for level         in _vars.levels_scheme.split('_')]
  learning_rate_list = [float(learning_rate) for learning_rate in _vars.learning_rate.split('_')]
  momentum_list      = [float(momentum     ) for momentum      in _vars.momentum     .split('_')] \
                       if _vars.momentum is not None else [None]*len(levels_list)

  print(f' Starting at level {levels_list[0]}')

  for k, (num_epochs, level, learning_rate, momentum) in enumerate(zip(
    num_epochs_list, levels_list, learning_rate_list, momentum_list,
  )):
    for g in _vars.optimizer.param_groups: g['lr'] = learning_rate

    if momentum is not None:
      for g in _vars.optimizer.param_groups: g['momentum'] = momentum

    # print(f'Optimizer: {_vars.optimizer}\n')

    for epoch in range(num_epochs + 1):
      ## Training
      if epoch > 0:
        training_output = _vars.model.train_(
          num_batches=_vars.num_training_batches, 
          compute_accuracy=False, 
          print_times=False, 
          get_batch=lambda: get_batch('training'), 
          **filter_keys(_vars.__dict__, ('model',)),
        )

      ## Evaluation
      validation_output = _vars.model.evaluate(
        num_batches=_vars.num_validation_batches, 
        compute_accuracy=False, 
        print_times=False, 
        get_batch=lambda: get_batch('validation'), 
        **filter_keys(_vars.__dict__, ('model',)),
      )

      if epoch > 0: print(epoch, training_output, validation_output)
      else        : print(epoch,                  validation_output)

    if k != len(num_epochs_list) - 1:
      print(f' Changing from level {levels_list[k]} to level {levels_list[k+1]}')
  print('-> Done.\n')

  if _vars.generate:
    print('\n4. Generating text')
    generate_text(**_vars.__dict__)
    print('-> Done.\n')

if __name__ == '__main__': main()




