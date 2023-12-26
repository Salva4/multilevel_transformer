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
_vars.model_dimension = 8
_vars.num_heads = 2
_vars.dim_ff = 16
_vars.dropout = 0.
_vars.ode_solver = 'RK4'
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

  ## Fine model
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

  _vars.model_fine = _vars.model
  _vars.optimizer_fine = initialize_optimizer(**_vars.__dict__)

  ## Coarse model
  continuous_blocks_num_layers = [
    _vars.num_encoder_layers//2, _vars.num_decoder_layers//2,
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

  _vars.model_coarse = _vars.model
  _vars.optimizer_coarse = initialize_optimizer(**_vars.__dict__)

  # f = open('log1.txt', 'w'); f.close()
  # f = open('log2.txt', 'w'); f.close()

  # def log1(x):
  #   with open('log1.txt', 'a') as f:
  #     f.write(str(x) + '\n')
  # def log2(x):
  #   with open('log2.txt', 'a') as f:
  #     f.write(str(x) + '\n')

  for pf, pc in zip(
    _vars.model_fine  .precontinuous_block.parameters(),
    _vars.model_coarse.precontinuous_block.parameters(),
  ):
    pf.data = pc.data.clone()

  for i in range(len(_vars.model_coarse.continuous_blocks[0].ψ)):
    for pf, pc in zip(
      _vars.model_fine  .continuous_blocks[0].ψ[2*i].parameters(), 
      _vars.model_coarse.continuous_blocks[0].ψ[  i].parameters(),
    ):
      assert pf.shape == pc.shape
      pf.data = pc.data.clone()

  for i in range(len(_vars.model_coarse.continuous_blocks[0].ψ)):
    for pf, pc in zip(
      _vars.model_fine  .continuous_blocks[1].ψ[2*i].parameters(), 
      _vars.model_coarse.continuous_blocks[1].ψ[  i].parameters(),
    ):
      assert pf.shape == pc.shape
      pf.data = pc.data.clone()

  for pf, pc in zip(
    _vars.model_fine  .postcontinuous_block.parameters(),
    _vars.model_coarse.postcontinuous_block.parameters(),
  ):
    pf.data = pc.data.clone()

  _vars.criterion = nn.CrossEntropyLoss(
    ignore_index=_vars.target_vocabulary.pad_id,
  )

  print(f'3. Training models')

  _vars.splits = ['training', 'validation']
  _vars.data_loader_iterators = dict(zip(
    _vars.splits, [iter(_vars.data_loaders[split]) for split in _vars.splits],
  ))

  def get_batch(split):
    batch = next(_vars.data_loader_iterators[split], None)

    if batch is None:
      _vars.data_loader_iterators[split] = iter(_vars.data_loaders[split])
      batch = next(_vars.data_loader_iterators[split], None)
      if batch is None: 
        raise Exception(f'Length of {split} data loader is 0.')

    input, target = batch
    batch = (input, target)

    return batch

  batch = get_batch('training')
  input, target = batch

  output_fine = _vars.model_fine(
    input=input, target=target, criterion=_vars.criterion, level=1,
  )
  output_coarse = _vars.model_coarse(
    input=input, target=target, criterion=_vars.criterion, level=0,
  )
  loss_fine = output_fine['loss']
  loss_coarse = output_coarse['loss']

  loss_fine  .backward()
  loss_coarse.backward()
  _vars.optimizer_fine  .step()
  _vars.optimizer_coarse.step()

  batch = get_batch('training')
  input, target = batch

  output_fine = _vars.model_fine(
    input=input, target=target, criterion=_vars.criterion, level=1,
  )
  output_coarse = _vars.model_coarse(
    input=input, target=target, criterion=_vars.criterion, level=0,
  )
  loss_fine = output_fine['loss']
  loss_coarse = output_coarse['loss']

  print(loss_fine  .item())
  print(loss_coarse.item())



