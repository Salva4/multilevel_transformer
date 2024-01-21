
print('Importing packages...')
import copy
import sys
import time
import torch
import torch.nn as nn
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

def main():
  ## debug mode #######################
  if _vars.debug:
    _vars.batch_size = 2
    # _vars.continuous = True
    _vars.max_length = 10
    _vars.num_layers = 8#2
    _vars.T = float(_vars.num_layers)
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

  print('2. Building model')
  continuous_blocks_num_layers = [_vars.num_layers]
  _vars.model = Model(
    continuous_blocks_num_layers=continuous_blocks_num_layers,
    initialize_weights=False, **_vars.__dict__,
  )
  print('-> Done.\n')

  if _vars.continuous:
    print(' 2.1 Turning the model continuous')
    continuous_blocks_T = [_vars.T]
    _vars.model = ContinuousModel(
      continuous_blocks_T=continuous_blocks_T, **_vars.__dict__,
    )
    print(' -> Done.\n')

  _vars.optimizer = initialize_optimizer(**_vars.__dict__)
  _vars.criterion = nn.CrossEntropyLoss(ignore_index=_vars.pad_token_id)

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

    for epoch in range(num_epochs + 1):#tqdm.tqdm(range(num_epochs + 1)):
      t0_epoch = time.time()

      ## Training
      if epoch > 0:
        training_output = _vars.model.train_(
          num_batches=num_training_batches,
          compute_accuracy=True, 
          print_times=False,
          get_batch=lambda: get_batch('training'), 
          level=level,
          # ode_solver='Heun',
          **filter_keys(_vars.__dict__, ('model', 'ode_solver')),
        )

      ## Evaluation
      validation_output = _vars.model.evaluate(
        num_batches=num_validation_batches,
        compute_accuracy=True, 
        print_times=False,
        get_batch=lambda: get_batch('validation'), 
        level=level,
        # ode_solver='Heun',
        **filter_keys(_vars.__dict__, ('model', 'ode_solver')),
      )

      if epoch > 0: 
        print(f'Epoch: {epoch}')
        print(f'''  training loss: {training_output['loss']}, ''' \
            + f'''training accuracy: {training_output['accuracy']}''')
        print(f'''  validation loss: {validation_output['loss']}, ''' \
            + f'''validation accuracy: {validation_output['accuracy']}''')
      else: 
        print(f'Epoch: {epoch}')
        print(f'''  validation loss: {validation_output['loss']}, ''' \
            + f'''validation accuracy: {validation_output['accuracy']}''')

      print(f'Epoch time: {time.time() - t0_epoch}')

    if k != len(num_epochs_list) - 1:
      ## We assume that the changes from coarse to fine are of exactly 1 level
      old_level, new_level = levels_list[k : k+2]
      print(f' Changing from level {levels_list[k]} to level {levels_list[k+1]}')

      if old_level > new_level:
        assert old_level - new_level == 1, 'Changes upwards cannot jump more than one level.'
        print(f' Interpolating weights')
        _vars.model.interpolate_weights(
          fine_level=new_level,
          interpolation=_vars.multilevel_interpolation,
        )
        print(' -> Done.\n')

  print('-> Done.\n')

if __name__ == '__main__': main()




