import argparse
import numpy as np
import torch
import torch.nn as nn

from data import obtain_data
from model import DTransformer
from training import train

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--chunk_size', type=int, default=128)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--dim_feedforward', type=int, default=2048)
parser.add_argument('--dropout', type=float, default=.1)
parser.add_argument('--load', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--nmonitor', type=int, default=1000)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--save', action='store_true')
_vars = parser.parse_args()

def main():
  _vars.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  obtain_data(_vars)
  _vars.model = DTransformer(_vars)
  _vars.optimizer = torch.optim.Adam(_vars.model.parameters(), lr=_vars.lr)
  _vars.criterion = nn.CrossEntropyLoss(ignore_index=_vars.voc['<pad>'])
  train(_vars)

if __name__ == '__main__': main()
























