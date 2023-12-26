print('Using pipeline template')

def main(
  args, obtain_data, continuous_blocks_num_layers, continuous_blocks_T,
  get_batch_fn_generator, compute_accuracy=False, print_example=False, 
  src_decoding_function=None, tgt_decoding_function=None, print_times=False,
):
  print('Importing packages...')
  import copy
  import torch
  import torch.nn as nn
  import sys
  print('-> Done.\n')

  print('Importing local files...')
  sys.path.append('..')
  from model.model import Model
  from continuous_model.continuous_model import ContinuousModel
  from src_utils.filter_dict import filter_keys
  from src_utils.optimizer import initialize_optimizer
  print('-> Done.\n')

  _vars = copy.deepcopy(args)

  _vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Device: {_vars.device}\n')

  torch.manual_seed(_vars.seed)

  ## DATA
  print('1. Loading data')
  obtain_data(_vars)#tqdm.tqdm(obtain_data(_vars))
  print(f"Number of training batches: {  len(_vars.data_loaders['training'  ])}")
  print(f"Number of validation batches: {len(_vars.data_loaders['validation'])}")
  print('-> Done.\n')

  print('2. Building model')
  _vars.model = Model(
    continuous_blocks_num_layers=continuous_blocks_num_layers,
    initialize_weights=False, **_vars.__dict__,
  )
  print('-> Done.\n')

  if _vars.continuous:
    print(' 2.1 Turning the model continuous')
    _vars.model = ContinuousModel(
      continuous_blocks_T=continuous_blocks_T, **_vars.__dict__,
    )
    print(' -> Done.\n')

  _vars.optimizer = initialize_optimizer(**_vars.__dict__)
  _vars.criterion = nn.CrossEntropyLoss(
    ignore_index=(
      0#_vars.pad_token_id if 'pad_token_id' in _vars.__dict__ else -100
    )
  )

  print(f'3. Training models')

  _vars.splits = ['training', 'validation']
  get_batch = get_batch_fn_generator(_vars)

  num_epochs_list    = [  int(num_epochs   ) for num_epochs    in _vars.num_epochs   .split('_')]
  levels_list        = [  int(level        ) for level         in _vars.levels_scheme.split('_')]
  learning_rate_list = [float(learning_rate) for learning_rate in _vars.learning_rate.split('_')]
  momentum_list      = [float(momentum     ) for momentum      in _vars.momentum     .split('_')] \
                       if _vars.momentum is not None else [None]*len(levels_list)

  print(f' Starting at level {levels_list[0]}')

  assert 'data_loaders' in _vars.__dict__ or (
    _vars.num_training_batches is not None and \
    _vars.num_validation_batches is not None
  ), 'If no data loaders are used in this example, a number of training and validation batches must be specified by calling, e.g., "main.py --num_training_batches 100 --num_validation_batches 50".'

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
      ## Training
      if epoch > 0:
        training_output = _vars.model.train_(
          num_batches=num_training_batches,
          get_batch=lambda: get_batch('training'), 
          level=level,
          compute_accuracy=compute_accuracy, 
          print_example=print_example,
          src_decoding_function=src_decoding_function,
          tgt_decoding_function=tgt_decoding_function,
          print_times=print_times,
          **filter_keys(_vars.__dict__, ('model',)),
        )

      ## Evaluation
      validation_output = _vars.model.evaluate(
        num_batches=num_validation_batches,
        get_batch=lambda: get_batch('validation'), 
        level=level,
        compute_accuracy=compute_accuracy, 
        print_example=print_example,
        src_decoding_function=src_decoding_function,
        tgt_decoding_function=tgt_decoding_function,
        print_times=print_times,
        **filter_keys(_vars.__dict__, ('model',)),
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




