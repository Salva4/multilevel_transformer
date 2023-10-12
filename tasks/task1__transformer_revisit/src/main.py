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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)#64
parser.add_argument('--continuous', action='store_true')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--model_name', type=str, default='transformer') # Linear, Transformer
parser.add_argument('--models_dir', type=str, default=None)
parser.add_argument('--momentum', type=float, default=0.)
parser.add_argument('--N', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=1000000000)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--output_fn', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--solver', type=str, default='Forward Euler')
parser.add_argument('--T', type=float, default=None)
args = parser.parse_args()
# parser.add_argument('--init', type=str, required=True)#default='xavier')
# parser.add_argument('--pe', type=str, required=True)#default='torch')
# parser.add_argument('--interpol', type=str, required=True)#default='constant') <-- always 'linear' now (MGOPT): I, R

# def assert_arguments():
  # assert args.model.lower() in ['linear', 'transformer', 'continuoustransformer']
  # assert args.init.lower() in ['normal', 'xavier', 'none']
  # assert args.pe.lower() in ['torch', 'alternative']
  # assert args.optimizer in ['Adam', 'SGD']

  ## ML weights initialization
  # assert args.interpol.lower() in ['constant', 'linear']

  ## MGOPT
  # assert args.N%(2**(args.n_lvls - 1)) == 0

def obtain_ds_dl():
  train = DATA_PATH_TRAIN
  dev = DATA_PATH_DEV

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
  # assert_arguments()
  if args.T is None and args.continuous: args.T = args.N

  ## Time monitoring
  t0 = time.time()

  # print('INFO: 20221127_01_MGOPT: MG/OPT 1st ord. consistency - 2nd approach.')

  criterion = nn.CrossEntropyLoss(ignore_index=0)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'device {device}\n')

  ## DS
  print('1. Obtaining datasets and dataloaders')
  train_ds, eval_ds, train_dl, eval_dl \
    = tqdm.tqdm(obtain_ds_dl())
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
  torch.manual_seed(0)

  model_architecture_path = '.'.join(
    ['model', 'architectures', args.model_name, 'architecture']
  )
  model = Model(
    model_architecture_path=model_architecture_path,
    N=args.N,
  ).to(device)

  if args.continuous:
    model = ContinuousModel(
      model=model,
      T=args.T,
      solver=args.solver,
    )

  # optimizer = (torch.optim.Adam if args.optimizer == 'Adam' else torch.optim.SGD)(model.parameters(), lr=args.lr)
  if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                momentum=args.momentum)#.9)
  else: raise Exception()

  print(f'model: {model}')
  print(f'optimizer: {optimizer}')
  # print(args.init.capitalize(), args.pe.capitalize(), args.T, args.N)
  ########################################

  print()
  torch.manual_seed(1)
  print(f'3. Training models')
  for epoch in tqdm.tqdm(range(args.num_epochs)):
    model, va_acc = train_epoch(train_dl, eval_dl, model, optimizer, 
                                criterion, device)
    print(f'Epoch {str(epoch).zfill(2)}\tVa acc:\t{va_acc : .4f}')

  ########################################
  
  print(f'Execution finished. Time: {time.time() - t0 : .2f}')

if __name__ == '__main__':
  main()








































