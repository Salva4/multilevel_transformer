# Author: Marc Salvadó Benasco

print('Importing modules...')#, end=' ')
import argparse
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers.models.marian.modeling_marian import MarianMTModel
import sys
print('-> Done.')

print('Importing local files...')#, end=' ')
sys.path.append('../../../src/')
from model.model import Model
from continuous_model.continuous_model import ContinuousModel

from argument_parsing import parse_arguments, assert_and_correct_arguments
from data import obtain_data
from task_utils import copy_weights
from train import evaluate_bleu
from generation import generate
print('-> Done.')

print('Parsing arguments...')#, end=' ')
args = parse_arguments()
assert_and_correct_arguments(args)
print('-> Done.')
print(f'args: {args}')

_vars = copy.deepcopy(args)
# _vars.debug = True
# _vars.model_dimension = 8
# _vars.num_heads = 2
# _vars.dim_ff = 16

# def main():
_vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {_vars.device}')

torch.manual_seed(0)

_vars.name_model = "Helsinki-NLP/opus-mt-en-de"
print('Loading tokenizer...', end=' ')
_vars.tokenizer = AutoTokenizer.from_pretrained(_vars.name_model)
print('-> Done.')

_vars.pad_token_id = _vars.tokenizer.pad_token_id
_vars.bos_token_id = _vars.pad_token_id
_vars.eos_token_id = _vars.tokenizer.eos_token_id

print('Loading data...')
obtain_data(_vars)
print(f"Number of batches: " \
    + f"train, {len(_vars.dl['train'])}; test, {len(_vars.dl['test'])}.")
print('-> Done.')

print('Loading pre-trained model...')
_vars.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(
  _vars.name_model,
).to(_vars.device)
print('-> Done.')

continuous_blocks_num_layers = [
  _vars.num_encoder_layers, _vars.num_decoder_layers,
]
_vars.model = Model(
  continuous_blocks_num_layers=continuous_blocks_num_layers, **_vars.__dict__,
)#Transformer(_vars)
copy_weights(_vars.pretrained_model, _vars.model)

# self, model = _vars.model, _vars.pretrained_model

# _vars.model.generate = \
#   lambda *args, **kwargs: generate(_vars.model, *args, **kwargs)
# _vars.model.generate = generate
_vars.model.generate = lambda *args, **kwargs: generate(*args, **kwargs)

## Debug forward pass ##################
# instance = next(iter(_vars.dl['train']))
# src = instance['input_ids'].to(_vars.device)
# tgt = instance['labels'   ].to(_vars.device)
# print(_vars.__dict__)
# model_inputs = {
#   'input': src, 'target': tgt, 
#   'criterion': nn.CrossEntropyLoss(ignore_index=58100),
# }
# outputs_model = _vars.model(**model_inputs)
# print(outputs_model)
# sys.exit()
########################################

## Debug generation ####################
# instance = next(iter(_vars.dl['train']))
# src = instance['input_ids'].to(_vars.device)
# print(_vars.__dict__)
# outputs_model = _vars.model.generate(
#   src=src,
#   max_new_tokens=40, 
#   do_sample=False,#True, 
#   top_k=30, 
#   top_p=0.95,
#   **_vars.__dict__,
# )
# outputs_pretrained = _vars.pretrained_model.generate(
#   src,
#   max_new_tokens=40, 
#   do_sample=False,#True, 
#   top_k=30, 
#   top_p=0.95
# )
# print(outputs_pretrained)
# print(outputs_model)
# print(f'outputs_pretrained.shape {outputs_pretrained.shape}, ' \
#     + f'outputs_model.shape {outputs_model.shape}')
# print(torch.eq(outputs_pretrained, outputs_model).all().item())
# sys.exit()
########################################

_vars.loss_function = nn.CrossEntropyLoss(ignore_index=_vars.pad_token_id)
_vars.optimizer = torch.optim.Adam(_vars.model.parameters(), lr=_vars.lr)

torch.manual_seed(1)

# for epoch in range(_vars.num_epochs)
  # train_epoch(_vars)

print('Evaluating bleu')
evaluate_bleu(_vars)
print(_vars.candidate_corpus)
print(_vars.reference_corpus)
print(_vars.bleu)


# if __name__ == '__main__': main()


