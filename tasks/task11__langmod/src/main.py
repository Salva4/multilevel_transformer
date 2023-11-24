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
from utils import get_batch
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

def obtain_model_name(_vars):
  # model_name = '_'.join([
  #   f'{k}{_vars.__dict__[k]}'.replace(' ', '_') \
  #   for k in sorted(_vars.__dict__.keys())
  # ]) # str(_vars)
  model_name = ''
  for (k, v) in sorted(_vars.__dict__.items()):
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

# def main():
if 1:
  # hyperparameters
  eval_interval = 1000#500
  learning_rate = float(_vars.learning_rate)
  _vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'device {_vars.device}')
  eval_iters = 200
  dropout = 0.#0.2
  # ------------

  if _vars.debug:
    _vars.batch_size = 2
    _vars.context_window = 5
    eval_interval = 3
    eval_iters = 1
    _vars.max_new_tokens = 10
    _vars.model_dimension = 8#32
    _vars.num_heads = 4
    # _vars.continuous = True

  print(_vars)

  torch.manual_seed(_vars.seed)

  ## DATA
  print('\n1. Loading data')
  # train_data, val_data, decode, vocabulary_size = \
  #   data.tokenize_data_at_character_level(DATA_PATH)
  train_data, val_data, decode, vocabulary_size = \
    data.obtain_data(
      DATA_DIR, _vars.input_text, _vars.tokenization, _vars.debug,
  )

  _vars.vocabulary_size = vocabulary_size
  _vars.dropout = dropout

  ## MODEL
  print('\n2. Building model')
  print(f'Building model w/ {_vars.num_layers} decoder layers')
  continuous_blocks_num_layers = [_vars.num_layers]
  _vars.model = Model(
    continuous_blocks_num_layers=continuous_blocks_num_layers, 
    initialize_weights=False, **_vars.__dict__,
  )

  torch.manual_seed(_vars.seed)
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
  get_training_set_batch = lambda: get_batch(
    'train', train_data, val_data, _vars.context_window, _vars.batch_size,
    _vars.device,
  )
  get_validation_set_batch = lambda: get_batch(
    'val', train_data, val_data, _vars.context_window, _vars.batch_size,
    _vars.device,
  )

  num_epochs_list = [
    int(num_epochs) for num_epochs in _vars.num_epochs.split('-')
  ]
  for num_epochs in num_epochs_list:
    for epoch in range(num_epochs + 1):
      # torch.manual_seed(batch_idx)

      ## Train
      if epoch > 0:
        output_train = _vars.model.train_(
          num_batches=1000, compute_accuracy=False, print_times=False, 
          get_batch=get_training_set_batch, 
          # gradient_accumulation_size=10, clipping_norm=.1,
          **filter_keys(_vars.__dict__, (
            'num_batches', 'compute_accuracy', 'print_times', 'get_batch',
            'model',
          )),
        )

      ## Evaluate
      output_test = _vars.model.evaluate(
        num_batches=eval_interval, compute_accuracy=False, print_times=False, 
        get_batch=get_validation_set_batch, 
        **filter_keys(_vars.__dict__, (
          'num_batches', 'compute_accuracy', 'print_times', 'get_batch',
          'model',
        )),
      )

      if epoch > 0: print(epoch, output_train, output_test)
      else: print(epoch, output_test)

  if _vars.generate:
    print('\n4. Generating text')
    # generate from the model
    generate_text(_vars.model, decode, **_vars.__dict__)

# if __name__ == '__main__':
#   main()


