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

def main():
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
      + f"train {len(_vars.data_loaders['train'])}, " \
      + f"test, {len(_vars.data_loaders['test'])}.")
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
    continuous_blocks_num_layers=continuous_blocks_num_layers, 
    initialize_weights=True, **_vars.__dict__,
  )#Transformer(_vars)
  # copy_weights(_vars.pretrained_model, _vars.model)

  # print(_vars.pretrained_model)
  # print(_vars.model)

  # for p in _vars.pretrained_model.parameters(): print(p.shape, p.ravel()[:5])
  # print('change')
  # for p in _vars.           model.parameters(): print(p.shape, p.ravel()[:5])
  # sys.exit()

  # self, model = _vars.model, _vars.pretrained_model

  _vars.model.generate = lambda *args, **kwargs: generate(*args, **kwargs)

  # ## Debug forward pass ##################
  # instance = next(iter(_vars.data_loaders['train']))
  # src = instance['input_ids'].to(_vars.device)
  # tgt = instance['labels'   ].to(_vars.device)
  # model_inputs = {
  #   'input': src, 'target': tgt, 
  #   'criterion': nn.CrossEntropyLoss(ignore_index=58100),
  #   'store_hidden_states': True,
  # }
  # outputs_model = _vars.model(**model_inputs)#['x']
  # outputs_pretrained = _vars.pretrained_model(
  #   input_ids=src, decoder_input_ids=tgt[:, :-1],
  # )#['logits']
  # import sys; sys.exit()

  # ## Conventional transformer
  # import numpy as np
  # class ConvTrans(nn.Module):
  #   def __init__(
  #     self, model_dimension, num_heads, num_encoder_layers, num_decoder_layers,
  #     dim_ff, device, tokenizer, pad_token_id, **kwargs,
  # ):
  #     super().__init__()
  #     self.model_dimension = model_dimension
  #     self.device = device

  #     self.embedding = nn.Embedding(
  #       len(tokenizer), model_dimension, padding_idx=pad_token_id,
  #     )
  #     self.positional_encoding_src = nn.Embedding(512, model_dimension)
  #     self.positional_encoding_tgt = nn.Embedding(512, model_dimension)

  #     self.transformer = nn.Transformer(
  #       d_model=_vars.model_dimension, 
  #       nhead=_vars.num_heads, 
  #       num_encoder_layers=_vars.num_encoder_layers, 
  #       num_decoder_layers=_vars.num_decoder_layers,
  #       dim_feedforward=_vars.dim_ff,
  #       dropout=0.,
  #       norm_first=True,
  #       device=_vars.device,
  #       batch_first=True,
  #     )
  #     self.classifier = nn.Linear(model_dimension, len(tokenizer))

  #   def forward(self, input, target, **kwargs):
  #     src, tgt = input, target[:, :-1]
  #     # mask_pad_src = torch.where(src.eq(self.pad_token_id), -np.inf, 0)  # mask_pad_src: [b, L]
  #     # mask_pad_mem = mask_pad_src                                  # mask_pad_mem: [b, L]
  #     # mask_pad_tgt = torch.where(tgt.eq(self.pad_token_id), -np.inf, 0)

  #     ## Embedding
  #     x = self.embedding(src)  # src: [b, L , d]
  #     y = self.embedding(tgt)  # tgt: [b, L', d]

  #     ## Scaling
  #     x *= np.sqrt(self.model_dimension)
  #     y *= np.sqrt(self.model_dimension)

  #     ## Positional encoding
  #     L, Lp = x.shape[1], y.shape[1]
  #     positions_src = torch.arange(L ).reshape(1, L ).to(self.device)  # positions_src: [1, L ]
  #     positions_tgt = torch.arange(Lp).reshape(1, Lp).to(self.device)  # positions_tgt: [1, L ]
  #     positional_encoding_src = self.positional_encoding_src(positions_src)  # positions_src: [1, L , d]
  #     positional_encoding_tgt = self.positional_encoding_tgt(positions_tgt)  # positions_src: [1, L , d]

  #     x += positional_encoding_src  # src: [b, L , d]
  #     y += positional_encoding_tgt  # tgt: [b, L , d]

  #     y = self.transformer(src=x, tgt=y)

  #     output = self.classifier(y)

  #     # print(output.shape, target[: 1:].shape)

  #     loss = nn.CrossEntropyLoss(ignore_index=58100)(
  #       output.transpose(1,2), target[:, 1:],
  #     )

  #     return {'loss': loss}#output

  # import numpy as np
  # import sys; sys.path.append('model_architectures/transformer/transformer_utils')
  # from F_enc import F_enc
  # from F_dec import F_dec
  # class ConvTrans(nn.Module):
  #   def __init__(
  #     self, model_dimension, num_heads, num_encoder_layers, num_decoder_layers,
  #     dim_ff, device, tokenizer, pad_token_id, criterion, **kwargs,
  # ):
  #     super().__init__()
  #     self.model_dimension = model_dimension  # aka 'd'
  #     self.device = device
  #     self.pad_token_id = pad_token_id
  #     self.criterion = criterion

  #     self.embedding = nn.Embedding(
  #       len(tokenizer),
  #       model_dimension,
  #       padding_idx=pad_token_id,
  #     )
  #     self.positional_encoding_src = nn.Embedding(512, model_dimension)
  #     self.positional_encoding_tgt = nn.Embedding(512, model_dimension)

  #     self.F_encs = nn.ModuleList([
  #         F_enc(model_dimension, num_heads, dim_ff) \
  #         for _ in range(num_encoder_layers)
  #     ])
  #     self.F_decs = nn.ModuleList([
  #         F_dec(model_dimension, num_heads, dim_ff) \
  #         for _ in range(num_decoder_layers)
  #     ])
  #     self.classifier = nn.Linear(model_dimension, len(tokenizer))

  #   def forward(self, input, target, **kwargs):
  #     src, tgt, labels = input, target[:, :-1], target[:, 1:]

  #     mask_pad_src = torch.where(src.eq(self.pad_token_id), -np.inf, 0)  # mask_pad_src: [b, L ]
  #     mask_pad_tgt = torch.where(tgt.eq(self.pad_token_id), -np.inf, 0)  # mask_pad_tgt: [b, L']
  #     mask_pad_mem = mask_pad_src                                        # mask_pad_mem: [b, L ]

  #     ## Embedding
  #     x = self.embedding(src)  # src: [b, L , d]
  #     y = self.embedding(tgt)  # tgt: [b, L', d]

  #     ## Scaling
  #     x *= np.sqrt(self.model_dimension)
  #     y *= np.sqrt(self.model_dimension)

  #     ## Positional encoding
  #     L, Lp = x.shape[1], y.shape[1]
  #     positions_src = torch.arange(L ).reshape(1, L ).to(self.device)  # positions_src: [1, L ]
  #     positions_tgt = torch.arange(Lp).reshape(1, Lp).to(self.device)  # positions_tgt: [1, L ]
  #     positional_encoding_src = self.positional_encoding_src(positions_src)  # positions_src: [1, L , d]
  #     positional_encoding_tgt = self.positional_encoding_tgt(positions_tgt)  # positions_src: [1, L , d]

  #     x += positional_encoding_src  # src: [b, L , d]
  #     y += positional_encoding_tgt  # tgt: [b, L , d]

  #     # print(f'x {x.ravel()[:5]}')
  #     # print(f'y {y.ravel()[:5]}')
  #     # import sys; sys.exit()

  #     for F in self.F_encs:
  #       x = x + F(x, mask_pad_src)
  #       # x = F(x=x, mask_pad_src=mask_pad_src)

  #     for F in self.F_decs:
  #       y = y + F(y, x, mask_pad_tgt, mask_pad_mem)
  #       # y = F(x=y, memory=x, mask_pad_tgt=mask_pad_tgt, mask_pad_mem=mask_pad_mem)

  #     output = self.classifier(y)

  #     # print(output.shape, target[: 1:].shape)

  #     loss = self.criterion(output.transpose(1,2), labels)

  #     return {'loss': loss}#output

  # _vars.criterion = nn.CrossEntropyLoss(ignore_index=_vars.pad_token_id)
  # conv_trans = ConvTrans(**_vars.__dict__).to(_vars.device)

  # outputs_conv = conv_trans(src, tgt[:, :-1])
  # print(outputs_pretrained.ravel()[:10])
  # print(outputs_model.ravel()[:10])
  # print(outputs_conv .ravel()[:10])
  # print(f'outputs_pretrained.shape {outputs_pretrained.shape}, ' \
  #     + f'outputs_model.shape {outputs_model.shape}')
  # print(torch.eq(outputs_pretrained, outputs_model).all().item())
  # sys.exit()
  ########################################

  ## Debug generation ####################
  # instance = next(iter(_vars.data_loaders['train']))
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

  # torch.manual_seed(1)
  # for p in _vars.model.parameters(): 
  #   if p.dim() > 1: 
  #     # print('pold', p.ravel()[:5])
  #     nn.init.xavier_uniform_(p)
  #     # print('pnew', p.ravel()[:5])
  #   else: p.data.normal_()
  # torch.manual_seed(1)
  # for p in conv_trans.parameters(): 
  #   if p.dim() > 1:
  #     # print('pold', p.ravel()[:5])
  #     nn.init.xavier_uniform_(p)
  #     # print('pnew', p.ravel()[:5])
  #   else: p.data.normal_()
  # for p in _vars.model.parameters(): print(p.shape, p.ravel()[:4])
  # for p in conv_trans.parameters(): print(p.shape, p.ravel()[:4])
  # sys.exit()

  _vars.criterion = nn.CrossEntropyLoss(ignore_index=_vars.pad_token_id)
  _vars.optimizer = torch.optim.Adam(_vars.model.parameters(), lr=_vars.lr)
  # _vars.optimizer = torch.optim.Adam(conv_trans .parameters(), lr=_vars.lr)

  _vars.data_loader_iterators = dict(zip(
    _vars.splits, [iter(_vars.data_loaders[split]) for split in _vars.splits],
  ))

  def get_batch(split):
    batch = next(_vars.data_loader_iterators[split], None)

    if batch is None: 
      _vars.data_loader_iterators[split] = iter(_vars.data_loaders[split])
      batch = next(_vars.data_loader_iterators[split], None)

    input, target = batch['input_ids'], batch['labels']
    batch = (input, target)

    # print(f'input {input.ravel()[:3]} target {target.ravel()[:3]}')

    return batch

  num_epochs_list = [
    int(num_epochs) for num_epochs in _vars.num_epochs.split('-')
  ]

  ## Training
  _vars_training_dict = _vars.__dict__.copy()
  _ = _vars_training_dict.pop('model')
  # print(next(_vars.model.continuous_blocks[1].layers[0].residual_layer.parameters()))
  # print(next(conv_trans.F_decs[0].parameters()))
  for num_epochs in num_epochs_list:
    for epoch in range(num_epochs + 1):
      # torch.manual_seed(1)
      _vars.data_loader_iterators['train'] = iter(_vars.data_loaders['train'])

      ## Prints
      # print(_vars.model.precontinuous_block.embedding.weight.data.ravel()[:10])
      # print(_vars.model.precontinuous_block.positional_encoding_src.weight.data.ravel()[:10])
      # print(_vars.model.precontinuous_block.positional_encoding_tgt.weight.data.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[-1].residual_layer.F.self_attn.attn.k_proj.weight.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[-1].residual_layer.F.self_attn.attn.k_proj.bias.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[-1].residual_layer.F.self_attn.attn.v_proj.weight.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[-1].residual_layer.F.self_attn.attn.v_proj.bias.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[-1].residual_layer.F.self_attn.attn.q_proj.weight.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[-1].residual_layer.F.self_attn.attn.q_proj.bias.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[-1].residual_layer.F.self_attn.attn.out_proj.weight.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[-1].residual_layer.F.self_attn.attn.out_proj.bias.ravel()[:10])
      # print(_vars.model.postcontinuous_block.classifier.weight.ravel()[:10])
      # print(_vars.model.postcontinuous_block.classifier.bias.ravel()[:10])
      # print()
      # print(_vars.model.continuous_blocks[0].layers[0].residual_layer.F.self_attn.attn.k_proj.weight.ravel()[:10])
      # print(_vars.model.continuous_blocks[0].layers[1].residual_layer.F.self_attn.attn.k_proj.weight.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[0].residual_layer.F.self_attn.attn.k_proj.weight.ravel()[:10])
      # print(_vars.model.continuous_blocks[1].layers[1].residual_layer.F.self_attn.attn.k_proj.weight.ravel()[:10])
      # print()

      # for F in conv_trans.F_encs:
      #   print(F.mlp.fc1.weight.ravel()[:10])
      #   print(F.mlp.fc2.bias.ravel()[:10])
      # print(conv_trans.classifier.weight.ravel()[:10])
      # print(conv_trans.classifier.bias.ravel()[:10])
      # print()

      if epoch > 0:
        output_train = _vars.model.train_(
          num_batches=100, compute_accuracy=False, print_times=False, 
          get_batch=lambda: get_batch('train'), 
          # gradient_accumulation_size=10, clipping_norm=.1,
          **_vars_training_dict,
        )
        # output_train = _vars.model.static_train(conv_trans,
        #   num_batches=100, compute_accuracy=False, print_times=False, 
        #   get_batch=lambda: get_batch('train'), 
        #   # gradient_accumulation_size=10, clipping_norm=.1,
        #   **_vars_training_dict,
        # )
      output_test = _vars.model.evaluate(
        num_batches=100, compute_accuracy=False, print_times=False, 
        get_batch=lambda: get_batch('test' ), **_vars_training_dict,
      )
      # output_test = _vars.model.static_evaluate(conv_trans,
      #   num_batches=100, compute_accuracy=False, print_times=False, 
      #   get_batch=lambda: get_batch('test'), **_vars_training_dict,
      # )

      if epoch > 0: print(epoch, output_train, output_test)
      else: print(epoch, output_test)

  # torch.manual_seed(1)

  # # for epoch in range(_vars.num_epochs)
  #   # train_epoch(_vars)

  # print('Evaluating bleu')
  # evaluate_bleu(_vars)
  # print(_vars.candidate_corpus)
  # print(_vars.reference_corpus)
  # print(_vars.bleu)


if __name__ == '__main__': main()




