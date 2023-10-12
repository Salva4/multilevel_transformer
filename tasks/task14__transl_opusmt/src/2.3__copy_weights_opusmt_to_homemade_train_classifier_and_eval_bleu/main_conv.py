# Author: Marc Salvadó Benasco
print('starting')
import argparse
import copy
import importlib
import math
# import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
# from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

from data import obtain_data
from model.transformer import Transformer
from train import train_epoch, evaluate_bleu

r = importlib.reload
# r(data)
# r(model)
# r(train)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--d', type=int, default=512)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--dim_ff', type=int, default=2048)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--num_layers_decoder', type=int, default=6)
parser.add_argument('--num_layers_encoder', type=int, default=6)
parser.add_argument('--norm_first', type=bool, default=False)
args = parser.parse_args()
_vars = copy.deepcopy(args)

_vars.debug = True

# def main():
_vars.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(_vars.dev)

torch.manual_seed(0)

_vars.name_model = "Helsinki-NLP/opus-mt-en-de"
_vars.tokenizer = AutoTokenizer.from_pretrained(_vars.name_model)
_vars.pad_id = _vars.tokenizer.pad_token_id
_vars.bos_id = _vars.pad_id
_vars.eos_id = _vars.tokenizer.eos_token_id

obtain_data(_vars)

_vars.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(_vars.name_model)
_vars.model = Transformer(_vars)
_vars.model.copy_weights(_vars.pretrained_model)

self, model = _vars.model, _vars.pretrained_model

## Check forward pass is the same
x = next(iter(_vars.dl['train']))
src = x['input_ids']
tgt = x['labels']
# torch.manual_seed(0)
# output_model = _vars.model(src, tgt)
# torch.manual_seed(0)
# output_pretrained = _vars.pretrained_model(
#   src, 
#   decoder_input_ids=tgt, 
#   attention_mask=src.ne(_vars.pad_id),
#   decoder_attention_mask=tgt.ne(_vars.pad_id),
# )
# print((output_model - output_pretrained['logits']).abs().max())

#   print(f"Number of samples: train, {len(_vars.dl['train'])}; test, {len(_vars.dl['test'])}.")

#   _vars.loss_function = nn.CrossEntropyLoss(ignore_index=_vars.pad_id)
#   _vars.optimizer_pretrained = torch.optim.Adam(_vars.pretrained_model.parameters(), lr=_vars.lr)
#   _vars.optimizer_model = torch.optim.Adam(_vars.model.parameters(), lr=_vars.lr)

#   torch.manual_seed(1)

#   # for epoch in range(_vars.num_epochs)
#     # train_epoch(_vars)

#   evaluate_bleu(_vars=_vars, model=model)
#   print(_vars.bleu)
#   # print(_vars.candidate_corpus)
#   # print(_vars.reference_corpus)

#   evaluate_bleu(_vars=_vars, model=pretrained_model)
#   print(_vars.bleu)


# if __name__ == '__main__': main()


