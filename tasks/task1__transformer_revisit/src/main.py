import argparse
import numpy as np 
# import matplotlib.pyplot as plt
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import sys

sys.path.append('../../../src/')

import input_pipeline
import preprocessing
from model.model import Model
from continuous_model.continuous_model import ContinuousModel
from training import train_epoch

DATA_PATH_TRAIN = '../data/en_gum-ud-train.conllu.txt'
DATA_PATH_DEV = '../data/en_gum-ud-dev.conllu.txt'
DATA_PATH_TRAIN_DEBUG = '../data/en_gum-ud-train.conllu_debug.txt'
DATA_PATH_DEV_DEBUG = '../data/en_gum-ud-dev.conllu_debug.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)#64
parser.add_argument('--coarsening_factor', type=int, default=2)
parser.add_argument('--continuous', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--interpol', type=str, default='constant')  # <-- always 'linear' in MG/OPT: I, R
parser.add_argument('--levels_scheme', type=str, default='0', help='2_1_2_1_0_...')
parser.add_argument('--lr', type=str, default='1e-2', help='lrlvl0_lrlvl1_...')
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--model_name', type=str, default='transformer') # Linear, Transformer
parser.add_argument('--models_dir', type=str, default=None)
parser.add_argument('--momentum', type=str, default='.9', help='momentumlvl0_momentumlvl1_...')
parser.add_argument('--N', type=int, default=4)
parser.add_argument('--num_epochs', type=str, default='1000000000', help='10_10_10_10_10_...')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--output_fn', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--solver', type=str, default='Forward Euler')
parser.add_argument('--T', type=float, default=None)
args = parser.parse_args()
# parser.add_argument('--init', type=str, required=True)#default='xavier')
# parser.add_argument('--pe', type=str, required=True)#default='torch')

# ## Experiment for PC-cpu
# args.debug = True
# args.continuous = True
# args.levels_scheme = '1_0_1'
# args.lr = '1e-2_1e-3'
# args.momentum = '0._.9'
# args.num_epochs = '2_2_2'
# args.optimizer = 'SGD'

def assert_arguments(args):
  # ML weights initialization
  num_levels = len(args.lr.split('_'))
  assert num_levels == len(args.momentum.split('_'))

  if not args.continuous:
    assert num_levels == 1 and len(args.levels_scheme.split('_')) == 1 \
                           and len(args.num_epochs.split('_')) == 1
  else:
    if num_levels > 1:
      assert args.N // args.coarsening_factor ** (num_levels - 1) >  0 and \
             args.N %  args.coarsening_factor ** (num_levels - 1) == 0
    assert len(args.levels_scheme.split('_')) == len(args.num_epochs.split('_'))

  # MGOPT
  # assert args.N%(2**(args.n_lvls - 1)) == 0

def obtain_ds_dl():
  train = DATA_PATH_TRAIN if not args.debug else DATA_PATH_TRAIN_DEBUG
  dev = DATA_PATH_DEV if not args.debug else DATA_PATH_DEV_DEBUG
  print('train', train, 'dev', dev)

  vocabs = input_pipeline.create_vocabs(train)

  attributes_input = [input_pipeline.CoNLLAttributes.FORM]
  attributes_target = [input_pipeline.CoNLLAttributes.XPOS]

  train_ds, train_dl = preprocessing.obtain_dataset(
    filename=train, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=args.batch_size, 
    bucket_size=args.max_len,
  )
  eval_ds, eval_dl = preprocessing.obtain_dataset(
    filename=dev, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=args.batch_size,#187, 
    bucket_size=args.max_len,
  )

  return train_ds, eval_ds, train_dl, eval_dl

def main():
  ## args managing ####################
  if args.debug:
    args.batch_size = 2
    # args.continuous = True
    args.max_len = 10
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
  train_ds, eval_ds, train_dl, eval_dl = tqdm.tqdm(obtain_ds_dl())
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
  torch.manual_seed(args.seed)

  model_architecture_path = '.'.join(
    ['model_architectures', args.model_name, 'architecture']
  )
  model = Model(
    model_architecture_path=model_architecture_path, N=args.N,
  ).to(device)

  if args.continuous:
    num_levels = len(args.lr.split('_'))
    model = ContinuousModel(
      model=model, T=args.T, solver=args.solver, 
      coarsening_factor=args.coarsening_factor, #num_levels=num_levels,
    )

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
  torch.manual_seed(args.seed)
  num_epochs_list = [int(num_epochs) for num_epochs in args.num_epochs.split('_')]
  levels_list = [int(level) for level in args.levels_scheme.split('_')]
  lr_list = [float(lr) for lr in args.lr.split('_')]
  momentum_list = [float(momentum) for momentum in args.momentum.split('_')]

  print(f'Starting at level {levels_list[0]}')

  for k, (num_epochs, level) in enumerate(zip(num_epochs_list, levels_list)):
    lr = lr_list[level]
    momentum = momentum_list[level]

    for g in optimizer.param_groups: g['lr'] = lr

    if args.optimizer == 'SGD':
      for g in optimizer.param_groups: g['momentum'] = momentum

    # print('optimizer', optimizer)

    for epoch in tqdm.tqdm(range(num_epochs)):
      model, va_acc = train_epoch(
        train_dl, eval_dl, model, optimizer, criterion, device, level
      )

      print(f'Epoch {str(epoch).zfill(2)}\tVa acc:\t{va_acc : .4f}')

    if k != len(num_epochs_list) - 1: 
      print(f'Changing from level {levels_list[k]} to level {levels_list[k+1]}')

  ########################################
  
  print(f'Execution finished. Time: {time.time() - t0 : .2f}')

if __name__ == '__main__':
  main()



