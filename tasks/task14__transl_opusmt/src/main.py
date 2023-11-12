# Author: Marc Salvadó Benasco

print('Importing modules...', end=' ')
import argparse
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
print('Done.')

print('Importing local files...', end=' ')
sys.path.append('../../../src/')
from model.model import Model
from continuous_model.continuous_model import ContinuousModel

from argument_parsing import parse_arguments, assert_and_correct_arguments
from data import obtain_data
from train import evaluate_bleu
from task_utils import copy_weights
print('Done.')

print('Parsing arguments...', end=' ')
args = parse_arguments()
assert_and_correct_arguments(args)
print('Done.')
print(f'args: {args}')

_vars = copy.deepcopy(args)
# _vars.debug = True

# def main():
_vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {_vars.device}')

torch.manual_seed(0)

_vars.name_model = "Helsinki-NLP/opus-mt-en-de"
print('Loading tokenizer...', end=' ')
_vars.tokenizer = AutoTokenizer.from_pretrained(_vars.name_model)
print('Done.')

_vars.pad_id = _vars.tokenizer.pad_token_id
_vars.bos_id = _vars.pad_id
_vars.eos_id = _vars.tokenizer.eos_token_id

print('Loading data...')
obtain_data(_vars)
print(f"Number of batches: " \
    + f"train, {len(_vars.dl['train'])}; test, {len(_vars.dl['test'])}.")
print('Done.')

print('Loading pre-trained model...')
_vars.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(
  _vars.name_model,
)
print('Done.')

continuous_blocks_num_layers = [
  _vars.num_encoder_layers, _vars.num_decoder_layers,
]
_vars.model = Model(
  continuous_blocks_num_layers=continuous_blocks_num_layers, **_vars.__dict__,
)#Transformer(_vars)
copy_weights(_vars.pretrained_model, _vars.model)

self, model = _vars.model, _vars.pretrained_model
_vars.model.generate = _vars.pretrained_model.generate

_vars.loss_function = nn.CrossEntropyLoss(ignore_index=_vars.pad_id)
_vars.optimizer = torch.optim.Adam(_vars.model.parameters(), lr=_vars.lr)

torch.manual_seed(1)

# for epoch in range(_vars.num_epochs)
  # train_epoch(_vars)

print('Evaluating bleu')
evaluate_bleu(_vars)
print(_vars.bleu)
print(_vars.candidate_corpus)
print(_vars.reference_corpus)


# if __name__ == '__main__': main()


