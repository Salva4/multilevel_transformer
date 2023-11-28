
print('Importing modules...')#, end=' ')
import copy
import torch
import torch.nn as nn
# import tqdm
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
print('-> Done.\n')

# torch.set_default_dtype(torch.float64)

print('Parsing arguments...')#, end=' ')
args = parse_arguments()
assert_and_correct_arguments(args)
print('-> Done.\n')
print(f'Args: {args}')

_vars = copy.deepcopy(args)
# args.model_dimension = 8
# args.max_length = 5
# args.num_epochs = '2'
# args.debug = True

# ## Experiment for PC-cpu
# args.debug = True
# args.continuous = True
# args.levels_scheme = '1_0_1'
# args.lr = '1e-2_1e-3'
# args.momentum = '0._.9'
# args.num_epochs = '2_2_2'
# args.optimizer = 'SGD'

# def assert_arguments(args):
#   ## ML weights initialization
#   num_levels = len(args.lr.split('_'))
#   assert num_levels == len(args.momentum.split('_'))

#   if not args.continuous:
#     assert num_levels == 1 and len(args.levels_scheme.split('_')) == 1 \
#                            and len(args.num_epochs.split('_')) == 1
#     assert not args.mgopt and not args.mgrit
#   else:
#     if num_levels > 1:
#       assert args.N // args.coarsening_factor ** (num_levels - 1) >  0 and \
#              args.N %  args.coarsening_factor ** (num_levels - 1) == 0
#     assert len(args.levels_scheme.split('_')) == \
#            len(args.num_epochs   .split('_'))

#   ## MGRIT, MGOPT, ...
#   # assert not (args.mgrit and args.mgopt)

#   ## MGOPT
#   # assert args.N%(2**(args.n_lvls - 1)) == 0

#   also add Adam -> ~momentum !

def main():
  ## debug mode #######################
  if _vars.debug:
    _vars.batch_size = 2
    # _vars.continuous = True
    _vars.max_length = 10
    _vars.num_layers = 8#2
  #####################################

  # assert_arguments(_vars)

  _vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Device: {_vars.device}\n')

  torch.manual_seed(_vars.seed)

  ## DATA
  print('1. Loading data')
  obtain_data(_vars)#tqdm.tqdm(obtain_data(_vars))
  print(f"Number of training batches: {  len(_vars.data_loaders['training'  ])}")
  print(f"Number of validation batches: {len(_vars.data_loaders['validation'])}")
  print('-> Done.\n')

  ############## ML weights initialization
  # ## Init with fewer layers? Information is at N
  # Ns = args.N.split('-')
  # nums_epochs = args.num_epochs.split('-')
  # lr = args.lr
  # for i, N_str in enumerate(Ns):
  #   N = int(N_str)
  #   if i != 0:
  #     lr *= args.lr_factor

  #   ## Training setup 2/2
  #   model = Model(
  #     args.init.capitalize(),
  #     args.pe.capitalize(),
  #     T=args.T,
  #     N=N,
  #     # interpol=args.interpol.lower(),
  #   ).to(device)
  #   optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  #   ## Initialize fine model with coarse model
  #   if i != 0:
  #     # model.continuous_block.init_weights_from_model(coarse_model)
  #     model.init_weights_from_model(coarse_model)
  #   else:
  #     model.init_params()

  #   ## Training
  #   # num_epochs = args.num_epochs//len(Ns)
  #   num_epochs = int(nums_epochs[i])
  #   coarse_model = train(training_dataloader, validation_dataloader, model, optimizer,
  #     criterion, device, num_epochs, args.n_monitoring)

  #   print(f'Training finished for N={N}')
  ########################################

  ################################# MG/OPT
  # print(f'2. Initializing models')
  # models = []
  # optimizers = []
  # for lvl in tqdm.tqdm(range(args.n_lvls)):
  #   N = args.N // 2**(args.n_lvls - lvl - 1)  # From coarse to fine
  #   model = Model(
  #     init_method = 'None' if lvl != (args.n_lvls - 1) else args.init.capitalize(),
  #     encoding = args.pe.capitalize(),
  #     T = args.T,
  #     N = N,# + 1,    # ((main's N (MGOPT) is multiple of power of 2; model's N is (a power of 2) + 1)) <-- not anymore
  #   ).to(device)
  #   models.append(model)

  #   optimizer = (torch.optim.Adam if args.optimizer == 'Adam' else torch.optim.SGD)(model.parameters(), lr=args.lr)
  #   optimizers.append(optimizer)
  ########################################

  ################################# Conventional training
  print('2. Building model')
  continuous_blocks_num_layers = [_vars.num_layers]
  _vars.model = Model(
    continuous_blocks_num_layers=continuous_blocks_num_layers,
    initialize_weights=False, **_vars.__dict__,
  )#.to(_vars.device)
  print('-> Done.\n')

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
  _vars.criterion = nn.CrossEntropyLoss(ignore_index=_vars.pad_token_id)#0)
  ########################################

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

    input, target = batch
    batch = (input, target)

    return batch

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

    for epoch in range(num_epochs + 1):#tqdm.tqdm(range(num_epochs + 1)):
      ## Training
      if epoch > 0:
        training_output = _vars.model.train_(
          num_batches=len(_vars.data_loaders['training']),
          compute_accuracy=False, 
          print_times=False,
          get_batch=lambda: get_batch('training'), 
          **filter_keys(_vars.__dict__, ('model',)),
        )

      ## Evaluation
      evaluation_output = _vars.model.evaluate(
        num_batches=len(_vars.data_loaders['validation']),
        compute_accuracy=False, 
        print_times=False,
        get_batch=lambda: get_batch('validation'), 
        **filter_keys(_vars.__dict__, ('model',)),
      )

      if epoch > 0: print(epoch, training_output, evaluation_output)
      else        : print(epoch,                  evaluation_output)

    if k != len(num_epochs_list) - 1:
      print(f' Changing from level {levels_list[k]} to level {levels_list[k+1]}')
  print('-> Done.\n')
  ########################################

if __name__ == '__main__': main()




