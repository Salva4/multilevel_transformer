
print('Importing packages...')#, end=' ')
import copy
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer
# from transformers.models.marian.modeling_marian import MarianMTModel
print('-> Done.\n')

print('Importing local files...')#, end=' ')
sys.path.append('../../../src/')
from model.model import Model
from continuous_model.continuous_model import ContinuousModel
from src_utils.filter_dict import filter_keys
from src_utils.optimizer import initialize_optimizer

from argument_parsing import parse_arguments, assert_and_correct_arguments
from data import obtain_data
from generation import generate
print('-> Done.\n')

print('Parsing arguments...')#, end=' ')
args = parse_arguments()
assert_and_correct_arguments(args)
print('-> Done.\n')
print(f'Args: {args}')

_vars = copy.deepcopy(args)

def main():
  _vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Device: {_vars.device}\n')

  torch.manual_seed(args.seed)

  _vars.name_model = "Helsinki-NLP/opus-mt-en-de"
  print('Loading pre-trained tokenizer...', end=' ')
  _vars.tokenizer = AutoTokenizer.from_pretrained(_vars.name_model)
  print('-> Done.\n')

  _vars.pad_token_id = _vars.tokenizer.pad_token_id
  _vars.bos_token_id = _vars.pad_token_id
  _vars.eos_token_id = _vars.tokenizer.eos_token_id

  print('1. Loading data...')
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
    initialize_weights=True, **_vars.__dict__,
  )

  if _vars.continuous:
    print(' 2.1 Turning the model continuous')
    continuous_blocks_T = [_vars.encoder_T, _vars.decoder_T]
    _vars.model = ContinuousModel(
      continuous_blocks_T=continuous_blocks_T,
      is_encoder_decoder_transformer=True,
      **_vars.__dict__,
    )
    print(' -> Done.\n')

  _vars.model.generate = lambda *args, **kwargs: generate(*args, **kwargs)

  _vars.optimizer = initialize_optimizer(**_vars.__dict__)
  _vars.criterion = nn.CrossEntropyLoss(ignore_index=_vars.pad_token_id)

  print(f'''Number of model parameters: {
    sum(θ.numel() for θ in _vars.model.parameters())
  }''')

  print(f'3. Training models')

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

    input, target = batch['input_ids'], batch['labels']
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
    else len(_vars.data_loaders['training'])
  num_validation_batches = _vars.num_validation_batches \
    if _vars.num_validation_batches is not None \
    else len(_vars.data_loaders['validation'])

  for k, (num_epochs, level, learning_rate, momentum) in enumerate(zip(
    num_epochs_list, levels_list, learning_rate_list, momentum_list,
  )):
    ## Reset optimizer
    _vars.optimizer = initialize_optimizer(**_vars.__dict__)

    for g in _vars.optimizer.param_groups: g['lr'] = learning_rate

    if momentum is not None:
      for g in _vars.optimizer.param_groups: g['momentum'] = momentum

    # print(f'Optimizer: {_vars.optimizer}\n')

    for epoch in range(num_epochs + 1):
      t0_epoch = time.time()

      # ## Multi-fidelity weights initialization experiment 1/3
      # solver_change_epoch = 100
      # ode_solver = 'Forward Euler' if epoch < solver_change_epoch else 'RK4'
      # if epoch == solver_change_epoch:
      #   print(f'Changing ODE solver from FE to RK4')
      #   for continuous_block in _vars.model.continuous_blocks:
      #     for i in range(0, len(continuous_block.ψ) - 2, 2):  # initialize unused layers for RK4 (t_n + h/2) using the closest layers
      #       for θ_i, θ_ip1, θ_ip2 in zip(
      #         continuous_block.ψ[i  ].parameters(),
      #         continuous_block.ψ[i+1].parameters(),
      #         continuous_block.ψ[i+2].parameters(),
      #       ): θ_ip1.data = 1/2*(θ_i.data.clone() + θ_ip2.data.clone())
      #     for θ_last_but_1, θ_last in zip(  # plus layer at t_n using t_{n-1} + h/2
      #       continuous_block.ψ[-2].parameters(),
      #       continuous_block.ψ[-1].parameters(),
      #     ): θ_last.data = θ_last_but_1.data.clone()

      ## Training
      if epoch > 0:
        training_output = _vars.model.train_(
          num_batches=num_training_batches,#100,
          compute_accuracy=False,
          print_times=False,
          get_batch=lambda: get_batch('training'),
          level=level,
          # ode_solver=ode_solver,  # for multi-fidelity weights initialization experiment 2/3
          **filter_keys(_vars.__dict__, ('model', 'ode_solver',)),
        )

      ## Evaluation
      validation_output = _vars.model.evaluate(
        num_batches=num_validation_batches,#100,
        compute_accuracy=False,
        print_times=False,
        get_batch=lambda: get_batch('validation'),
        level=level,
        # ode_solver=ode_solver,  # for multi-fidelity weights initialization experiment 3/3
        **filter_keys(_vars.__dict__, ('model', 'ode_solver',)),
      )

      if epoch > 0:
        print(f'Epoch: {epoch}')
        print(f'''  training loss: {training_output['loss']}''')
        print(f'''  validation loss: {validation_output['loss']}''')
      else:
        print(f'Epoch: {epoch}')
        print(f'''  validation loss: {validation_output['loss']}, ''')

      print(f'Epoch time: {time.time() - t0_epoch}')

    if k != len(num_epochs_list) - 1:
      print(f' Changing from level {levels_list[k]} to level {levels_list[k+1]}')
  print('-> Done.\n')

if __name__ == '__main__': main()




