
## This code is adapted from my (Marc Salvadó Benasco) final project delivery
##...(Assignment 4) of the Deep Learning Lab course in in 2021, Autumn 
##...semester, at Università della Svizzera Italiana.

print('Importing packages...')
import copy
import torch
import torch.nn as nn
import sys
print('-> Done.\n')

print('Importing local files...')
sys.path.append('../../../src/')
from model.model import Model
from continuous_model.continuous_model import ContinuousModel
from src_utils.filter_dict import filter_keys
from src_utils.optimizer import initialize_optimizer

from argument_parsing import parse_arguments, assert_and_correct_arguments
from data import obtain_data
print('-> Done.\n')

print('Parsing arguments...')
args = parse_arguments()
assert_and_correct_arguments(args)
print('-> Done.\n')
print(f'Args: {args}')

_vars = copy.deepcopy(args)

# def main():
_vars.debug = True
_vars.continuous = True
if 1:
  _vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Device: {_vars.device}\n')

  torch.manual_seed(args.seed)
  
  ## DATA
  print('1. Loading data')
  obtain_data(_vars)
  print(f"Number of training batches: {  len(_vars.data_loaders['training'  ])}")
  print(f"Number of validation batches: {len(_vars.data_loaders['validation'])}")
  print('-> Done.\n')

  print('2. Building model')
  continuous_blocks_num_layers = [
    _vars.num_encoder_layers, _vars.num_decoder_layers,
  ]
  _vars.model = Model(
    continuous_blocks_num_layers=continuous_blocks_num_layers,
    initialize_weights=False, **_vars.__dict__,
  )
  print('-> Done.\n')

  if _vars.continuous:
    print(' 2.1 Turning the model continuous')
    continuous_blocks_T = [_vars.encoder_T, _vars.decoder_T]
    _vars.model = ContinuousModel(
      continuous_blocks_T=continuous_blocks_T,
      is_encoder_decoder_transformer=True,
      **_vars.__dict__,
    )
    print(' -> Done.\n')

  _vars.optimizer = initialize_optimizer(**_vars.__dict__)
  _vars.criterion = nn.CrossEntropyLoss(
    ignore_index=_vars.target_vocabulary.pad_id,
  )

  print(f'3. Training models')

  torch.manual_seed(0)

  _vars.splits = ['training', 'validation']
  _vars.data_loader_iterators = dict(zip(
    _vars.splits, [iter(_vars.data_loaders[split]) for split in _vars.splits],
  ))

  def get_batch(split):
    torch.manual_seed(0)
    batch = next(_vars.data_loader_iterators[split], None)

    if batch is None:
      _vars.data_loader_iterators[split] = iter(_vars.data_loaders[split])
      batch = next(_vars.data_loader_iterators[split], None)
      if batch is None: 
        raise Exception(f'Length of {split} data loader is 0.')

    input, target = batch
    batch = (input, target)

    return batch

  num_epochs_list    = [  int(num_epochs   ) for num_epochs    in _vars.num_epochs   .split('_')]
  levels_list        = [  int(level        ) for level         in _vars.levels_scheme.split('_')]
  learning_rate_list = [float(learning_rate) for learning_rate in _vars.learning_rate.split('_')]
  momentum_list      = [float(momentum     ) for momentum      in _vars.momentum     .split('_')] \
                       if _vars.momentum is not None else [None]*len(levels_list)

  print(f' Starting at level {levels_list[0]}')

  num_training_batches = _vars.num_training_batches \
    if _vars.num_training_batches is not None \
    else 500#len(_vars.data_loaders['training'])
  num_validation_batches = _vars.num_validation_batches \
    if _vars.num_validation_batches is not None \
    else 50#len(_vars.data_loaders['validation'])

  ## Decoding functions for example printing:
  src_decoding_function = lambda x: ''.join(
    [_vars.source_vocabulary.id_to_string[xi] for xi in x]
  )
  tgt_decoding_function = lambda x: ''.join(
    [_vars.target_vocabulary.id_to_string[xi] for xi in x]
  )

  print('\n*Attention*: Accuracy is computed w.r.t. whole sentences.\n')

  for k, (num_epochs, level, learning_rate, momentum) in enumerate(zip(
    num_epochs_list, levels_list, learning_rate_list, momentum_list,
  )):
    for g in _vars.optimizer.param_groups: g['lr'] = learning_rate

    if momentum is not None:
      for g in _vars.optimizer.param_groups: g['momentum'] = momentum


  ## Debug
  torch.manual_seed(0)
  num_training_batches = 2
  num_validation_batches = 3

    # print(f'Optimizer: {_vars.optimizer}\n')

    # for epoch in range(num_epochs + 1):#tqdm.tqdm(range(num_epochs + 1)):
      ## Training
      # if epoch > 0:

  _vars.gradient_accumulation_size = 1

  for _ in range(2):
    training_output = _vars.model.train_(
      num_batches=num_training_batches,
      get_batch=lambda: get_batch('training'),
      compute_accuracy='sentences',#True,
      print_example=True,
      src_decoding_function=src_decoding_function,
      tgt_decoding_function=tgt_decoding_function,
      print_times=False,
      level=level,
      **filter_keys(_vars.__dict__, ('model',)),
    )
    print(training_output)

    ## Evaluation
    validation_output = _vars.model.evaluate(
      num_batches=num_validation_batches,
      get_batch=lambda: get_batch('validation'),
      compute_accuracy='sentences',#True,
      print_example=True,
      src_decoding_function=src_decoding_function,
      tgt_decoding_function=tgt_decoding_function,
      print_times=False,
      level=level,
      **filter_keys(_vars.__dict__, ('model',)),
    )

    print(validation_output)

#       if epoch > 0: 
#         print(f'Epoch: {epoch}')
#         print(f'''  training_loss: {training_output['loss']}, ''' \
#             + f'''training_accuracy: {training_output['accuracy']*100}%''')
#         print(f'''  validation_loss: {validation_output['loss']}, ''' \
#             + f'''validation_accuracy: {validation_output['accuracy']*100}%''')
#       else: 
#         print(f'Epoch: {epoch}')
#         print(f'''  validation_loss: {validation_output['loss']}, ''' \
#             + f'''validation_accuracy: {validation_output['accuracy']*100}%''')

#     if k != len(num_epochs_list) - 1:
#       ## We assume that the changes from coarse to fine are of exactly 1 level
#       old_level, new_level = levels_list[k : k+2]
#       print(f' Changing from level {levels_list[k]} to level {levels_list[k+1]}')

#       if old_level > new_level:
#         assert old_level - new_level == 1, 'Changes upwards cannot jump more than one level.'
#         print(f' Interpolating weights')
#         _vars.model.interpolate_weights(
#           fine_level=new_level,
#           interpolation=_vars.multilevel_interpolation,
#         )
#         print(' -> Done.\n')

#   print('-> Done.\n')

# if __name__ == '__main__': main()




