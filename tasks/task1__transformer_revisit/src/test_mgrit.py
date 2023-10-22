# import argparse
import numpy as np 
# import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import sys

sys.path.append('../../../src/')

from argument_parsing import parse_arguments
import input_pipeline
import preprocessing
from model.model import Model
from continuous_model.continuous_model import ContinuousModel
from training import train_epoch

# torch.set_default_dtype(torch.float64)

TRAINING_DATA_PATH = '../data/en_gum-ud-train.conllu.txt'#'/users/msalvado/MLT/ML_PQ/data/en_gum-ud-train.conllu.txt'
VALIDATION_DATA_PATH = '../data/en_gum-ud-dev.conllu.txt'#'/users/msalvado/MLT/ML_PQ/data/en_gum-ud-dev.conllu.txt'
TRAINING_DATA_PATH_DEBUG = '../data/en_gum-ud-train.conllu_debug.txt'
VALIDATION_DATA_PATH_DEBUG = '../data/en_gum-ud-dev.conllu_debug.txt'

args = parse_arguments()
# args.model_dimension = 8
# args.max_len = 5
# args.num_epochs = '2'
# args.debug = True

# ## Experiment for PC-cpu
args.debug = True
args.continuous = True
# args.levels_scheme = '1_0_1'
# args.lr = '1e-2_1e-3'
# args.momentum = '0._.9'
# args.num_epochs = '2_2_2'
# args.optimizer = 'SGD'
args.mgrit = True

def assert_arguments(args):
  ## ML weights initialization
  num_levels = len(args.lr.split('_'))
  assert num_levels == len(args.momentum.split('_'))

  if not args.continuous:
    assert num_levels == 1 and len(args.levels_scheme.split('_')) == 1 \
                           and len(args.num_epochs.split('_')) == 1
    assert not args.mgopt and not args.mgrit
  else:
    if num_levels > 1:
      assert args.N // args.coarsening_factor ** (num_levels - 1) >  0 and \
             args.N %  args.coarsening_factor ** (num_levels - 1) == 0
    assert len(args.levels_scheme.split('_')) == \
           len(args.num_epochs   .split('_'))

  ## MGRIT, MGOPT, ...
  # assert not (args.mgrit and args.mgopt)

  ## MGOPT
  # assert args.N%(2**(args.n_lvls - 1)) == 0

def obtain_ds_dl():
  training_data_path   = TRAINING_DATA_PATH   if not args.debug else \
                         TRAINING_DATA_PATH_DEBUG
  validation_data_path = VALIDATION_DATA_PATH if not args.debug else \
                         VALIDATION_DATA_PATH_DEBUG
  print(
    'training_data_path'  , training_data_path, 
    'validation_data_path', validation_data_path,
  )

  vocabularies = input_pipeline.create_vocabularies(training_data_path)
  vocabulary_size = len(vocabularies['forms'])
  num_classes = len(vocabularies['xpos'])

  attributes_input = [input_pipeline.CoNLLAttributes.FORM]
  attributes_target = [input_pipeline.CoNLLAttributes.XPOS]

  training_dataset, training_dataloader = preprocessing.obtain_dataset(
    filename=training_data_path, 
    vocabularies=vocabularies, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=args.batch_size, 
    bucket_size=args.max_length,
    seed=0,
  )
  validation_dataset, validation_dataloader = preprocessing.obtain_dataset(
    filename=validation_data_path, 
    vocabularies=vocabularies, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=args.batch_size,#187, 
    bucket_size=args.max_length,
    seed=0,
  )

  return (
    training_dataset, validation_dataset, training_dataloader, 
    validation_dataloader, vocabulary_size, num_classes,
  )

def main():
  ## args managing ####################
  if args.debug:
    args.batch_size = 2
    # args.continuous = True
    args.max_length = 10
    args.N = 8#2

  assert_arguments(args)

  if args.T is None and args.continuous: args.T = args.N
  # args.solver = args.solver.replace('_', ' ')  # Forward_Euler --> Forward Euler

  print('args', args)
  #####################################


  ## Time monitoring
  t0 = time.time()

  # print('INFO: 20221127_01_MGOPT: MG/OPT 1st ord. consistency - 2nd approach.')

  criterion = nn.CrossEntropyLoss(ignore_index=0)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'device {device}\n')

  ## DS
  print('1. Obtaining datasets and dataloaders')
  (
    training_dataset, validation_dataset, training_dataloader, 
    validation_dataloader, vocabulary_size, num_classes,
  ) = tqdm.tqdm(obtain_ds_dl())
  print()

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
  #   coarse_model = train(train_dl, eval_dl, model, optimizer, 
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
  torch.manual_seed(0)#args.seed)

  model_architecture_path = '.'.join(
    ['model_architectures', args.model_name, 'architecture']
  )
  model = Model(
    model_architecture_path=model_architecture_path, 
    vocabulary_size=vocabulary_size, num_classes=num_classes, **args.__dict__,
  ).to(device)
  
  # for p in model.parameters():
  #   print(f'{p.dtype}, {p.shape}, {p.ravel()[:10]}')
  # sys.exit()

  if args.continuous:
    num_levels = len(args.lr.split('_'))
    model = ContinuousModel(model=model, **args.__dict__)#.to(device)

  # optimizer = (torch.optim.Adam if args.optimizer == 'Adam' else torch.optim.SGD)(model.parameters(), lr=args.lr)
  if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.)
  elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
      model.parameters(), lr=0., momentum=0.,
    )#.9)
  else: raise Exception()

  print(f'model: {model}')
  # print(f'optimizer: {optimizer}')
  # print(args.init.capitalize(), args.pe.capitalize(), args.T, args.N)
  ########################################

  print()
  print(f'3. Training models')
  torch.manual_seed(0)#args.seed)
  num_epochs_list = [int(num_epochs) for num_epochs in args.num_epochs.split('_')]
  levels_list = [int(level) for level in args.levels_scheme.split('_')]
  lr_list = [float(lr) for lr in args.lr.split('_')]
  momentum_list = [float(momentum) for momentum in args.momentum.split('_')]

  print(f'Starting at level {levels_list[0]}')

  level = 0
  lr = lr_list[level]
  momentum = momentum_list[level]

  for g in optimizer.param_groups: g['lr'] = lr

  if args.optimizer == 'SGD':
    for g in optimizer.param_groups: g['momentum'] = momentum

  model.train()
  batch = next(iter(training_dataloader))
  input_ids, target_ids = batch
  input_ids, target_ids = input_ids.to(device), target_ids.to(device)

  model_inputs = {
      'input': input_ids, 'target': target_ids, 'criterion': criterion, 
      'compute_accuracy': True,
    }
  model_inputs_conv  = model_inputs.copy()
  model_inputs_mgrit = model_inputs.copy()

  ## Conventional
  model_inputs_conv.update({'level': level})
  t0_conv = time.time()
  with torch.no_grad():
    model_outputs_conv = model(**model_inputs_conv)
  t_conv = time.time() - t0_conv

  ## MGRIT
  model_inputs_mgrit.update({
    'MGRIT': True, 'num_iterations': 3, 'num_levels': 2, 'relaxation': 'F',
  })
  t0_mgrit = time.time()
  with torch.no_grad():
    model_outputs_mgrit = model(**model_inputs_mgrit)
  t_mgrit = time.time() - t0_mgrit

  # print('  conv' , outputs      .ravel()[-10])
  # print('  mgrit', outputs_mgrit.ravel()[-10])

  loss_conv = model_outputs_conv['loss']
  correct, total = model_outputs_conv['correct'], model_outputs_conv['total']
  accuracy_conv = correct/total

  loss_mgrit = model_outputs_mgrit['loss']
  correct, total = model_outputs_mgrit['correct'], \
                   model_outputs_mgrit['total']
  accuracy_mgrit = correct/total

  print()
  print('loss:')
  print('  conv ', loss_conv .item())
  print('  mgrit', loss_mgrit.item())

  print()
  print('accuracy:')
  print('  conv ', accuracy_conv )
  print('  mgrit', accuracy_mgrit)

  print()
  print('time:')
  print('  conv ', t_conv )
  print('  mgrit', t_mgrit)

  ########################################
  
  print(f'Execution finished. Time: {time.time() - t0 : .2f}')

if __name__ == '__main__':
  main()



