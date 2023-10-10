import numpy as np 
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import input_pipeline
import preprocessing
from models import *    # Linear, Transformer
from continuous_block import *  # ContinuousTransformer
from training import train_epoch#train_MGOPT

import argparse
import time
import sys
import tqdm

## Colored output in terminal
# try:
#   from termcolor import colored
# except:
#   color = lambda z, col: print(z)
# else:
#   color = lambda z, col: print(colored(z), col)
color = lambda z, col: print(z)

sys.path.append('../utils')
import remove_undesired

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)#default=1e-2)
parser.add_argument('--optim', type=str, required=True)#default=1e-2)
parser.add_argument('--init', type=str, required=True)#default='xavier')
parser.add_argument('--pe', type=str, required=True)#default='torch')
parser.add_argument('--N', 
  type=int,#str, 
  required=True,
  help='''
- Previous implementation (ML for weights initialization): 
  string of ints joint by '-'; example: 3-5-9. 
  *comment1: N was the number of basis functions (i.e. points to interpolate) --> (power of 2) + 1
  *comment2: num_epochs had to correspond. example: '10-20-40'.
- Current implementation (MG/OPT):
  integer, must be multiple of a 2^(#lvls)
''',
)#default=4)
parser.add_argument('--T', type=float, required=True)#default=1.)
parser.add_argument('--num_epochs', 
  type=int,#str, 
  required=True,
  help='''
- Previous implementation (ML for weights initialization): <-- refer to --N
- Current implementation (MG/OPT): 
  integer
'''
  )#default=240)
# parser.add_argument('--interpol', type=str, required=True)#default='constant') <-- always 'linear' now (MGOPT): I, R
parser.add_argument('--batch_size', type=int, required=True)#default=64)
# parser.add_argument('--lr_factor', type=float, required=True)#default=1.)     <-- currently unused now (MGOPT). Same lr for all levels
# parser.add_argument('--n_monitoring', type=int, required=True)#default=10)
# parser.add_argument('--n_lvls', type=int, required=True)#default=2)
# parser.add_argument('--n_V_cycles', type=int, required=True)#default=1)
# parser.add_argument('--mus_nus', type=str, required=True,
#                     help="mu0_mu1-nu1_mu2-nu2_... where lvl0 is the coarsest")
# parser.add_argument('--lr_MGOPT', type=float, required=True,
#                     help="learning rate for interpolating during MG/OPT. If it is -1., it means 'avoid MGOPT'")
parser.add_argument('--output_fn', type=str, required=True,
          help="Saving model")
parser.add_argument('--models_dir', type=str, required=True,
          help="Saving model")
args = parser.parse_args()

data_path_train = '../data/en_gum-ud-train.conllu.txt'
data_path_dev = '../data/en_gum-ud-dev.conllu.txt'

model_name = args.model.capitalize()
if model_name in locals() \
  and str(locals()[model_name]) in [
    f"<class 'models.{model_name}'>",
    f"<class 'continuous_block.{model_name}'>"
  ]:
  exec(f'Model = {model_name}')
else:
  raise Exception('model name unknown')

batch_size = args.batch_size
max_len = 2048

def assert_arguments():
  assert args.model.lower() in ['linear', 'transformer', 'continuoustransformer']
  assert args.init.lower() in ['normal', 'xavier', 'none']
  assert args.pe.lower() in ['torch', 'alternative']
  assert args.optim in ['Adam', 'SGD']

  ## ML weights initialization
  # assert args.interpol.lower() in ['constant', 'linear']

  ## MGOPT
  # assert args.N%(2**(args.n_lvls - 1)) == 0

def obtain_ds_dl():
  train = data_path_train
  dev = data_path_dev

  vocabs = input_pipeline.create_vocabs(train)

  attributes_input = [input_pipeline.CoNLLAttributes.FORM]
  attributes_target = [input_pipeline.CoNLLAttributes.XPOS]

  train_ds, train_dl = preprocessing.obtain_dataset(
    filename=train, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=batch_size, 
    bucket_size=max_len,
  )
  eval_ds, eval_dl = preprocessing.obtain_dataset(
    filename=dev, 
    vocabs=vocabs, 
    attributes_input=attributes_input, 
    attributes_target=attributes_target,
    batch_size=batch_size,#187, 
    bucket_size=max_len,
  )

  return train_ds, eval_ds, train_dl, eval_dl

def main():
  assert_arguments()

  ## Time monitoring
  t0 = time.time()

  print('INFO: 20221127_01_MGOPT: MG/OPT 1st ord. consistency - 2nd approach.')

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

  #   optimizer = (torch.optim.Adam if args.optim == 'Adam' else torch.optim.SGD)(model.parameters(), lr=args.lr)
  #   optimizers.append(optimizer)
  ########################################

  ################################# Conventional training
  model = Model(
    init_method = args.init.capitalize(),
    encoding = args.pe.capitalize(), 
    T = args.T, 
    N = args.N,
  ).to(device)

  optimizer = (torch.optim.Adam if args.optim == 'Adam' else torch.optim.SGD)(model.parameters(), lr=args.lr)

  print(model)
  print(args.init.capitalize(), args.pe.capitalize(), args.T, args.N)
  print(optimizer)
  ########################################

  print()
  print(f'3. Training models')
  for epoch in tqdm.tqdm(range(args.num_epochs)):
    # model = train_MGOPT(train_dl, eval_dl, models, optimizers, criterion, device, #args.num_epochs, 
    #   args.n_monitoring, args.n_V_cycles, args.mus_nus, args.lr_MGOPT)
    model, va_acc = train_epoch(train_dl, eval_dl, model, optimizer, 
                                criterion, device)
    print(f'Epoch {str(epoch).zfill(2)}\tVa acc:\t{va_acc : .4f}')

  ########################################
  
  print(f'Execution finished. Time: {time.time() - t0 : .2f}')

if __name__ == '__main__':
  main()
  # remove_undesired.do()








































