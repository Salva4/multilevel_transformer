# Author: Marc Salvadó Benasco

import argparse
import copy
import math
# import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
# from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data import *
from train import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--d', type=int, default=512)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--dim_ff', type=int, default=2048)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--num_layers_decoder', type=int, default=6)
parser.add_argument('--num_layers_encoder', type=int, default=6)
parser.add_argument('--norm_first', type=bool, default=False)
args = parser.parse_args()
_vars = copy.deepcopy(args)


def main():
  _vars.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(_vars.dev)

  torch.manual_seed(0)

  _vars.name_model = "Helsinki-NLP/opus-mt-en-de"
  _vars.tokenizer = AutoTokenizer.from_pretrained(_vars.name_model)
  _vars.model = AutoModelForSeq2SeqLM.from_pretrained(_vars.name_model).to(_vars.dev)

  obtain_data(_vars)
  print(f"Number of samples: train, {len(_vars.dl['train'])}; test, {len(_vars.dl['test'])}.")

  _vars.pad_id = _vars.tokenizer.pad_token_id
  _vars.loss_function = nn.CrossEntropyLoss(ignore_index=_vars.pad_id)
  _vars.optimizer = torch.optim.Adam(_vars.model.parameters(), lr=_vars.lr)

  torch.manual_seed(1)

  # for epoch in range(_vars.num_epochs)
    # train_epoch(_vars)

  evaluate_bleu(_vars)
  print(_vars.bleu)
  print(_vars.candidate_corpus)
  print(_vars.reference_corpus)


if __name__ == '__main__': main()


