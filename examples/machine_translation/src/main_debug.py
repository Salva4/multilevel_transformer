
print('Importing modules...')#, end=' ')
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer
# from transformers.models.marian.modeling_marian import MarianMTModel
import sys
print('-> Done.\n')

print('Importing local files...')#, end=' ')
sys.path.append('../../../src/')
from model.model import Model
from continuous_model.continuous_model import ContinuousModel
from src_utils.filter_dict import filter_keys
from src_utils.optimizer import initialize_optimizer

from argument_parsing import parse_arguments, assert_and_correct_arguments
from data import obtain_data
from generation import generate
print('-> Done.\n')

print('Parsing arguments...')#, end=' ')
args = parse_arguments()
assert_and_correct_arguments(args)
print('-> Done.\n')
print(f'Args: {args}')

_vars = copy.deepcopy(args)
# _vars.debug = True
# _vars.model_dimension = 8
# _vars.num_heads = 2
# _vars.dim_ff = 16

def main():
  _vars.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Device: {_vars.device}\n')

  torch.manual_seed(args.seed)

  _vars.name_model = "Helsinki-NLP/opus-mt-en-de"
  print('Loading pre-trained tokenizer...', end=' ')
  _vars.tokenizer = AutoTokenizer.from_pretrained(_vars.name_model)
  print('-> Done.\n')

  _vars.pad_token_id = _vars.tokenizer.pad_token_id
  _vars.bos_token_id = _vars.pad_token_id
  _vars.eos_token_id = _vars.tokenizer.eos_token_id

  print('1. Loading data...')
  obtain_data(_vars)
  print(f"Number of training batches: {  len(_vars.data_loaders['training'  ])}")
  print(f"Number of validation batches: {len(_vars.data_loaders['validation'])}")
  print('-> Done.\n')

  print('2. Building model')
  continuous_blocks_num_layers = [
    _vars.num_encoder_layers, _vars.num_decoder_layers,
  ]
  _vars.model = Model(
    continuous_blocks_num_layers=continuous_blocks_num_layers, 
    initialize_weights=True, **_vars.__dict__,
  )

  if _vars.continuous:
    print(' 2.1 Turning the model continuous')
    continuous_blocks_T = [_vars.encoder_T, _vars.decoder_T]
    _vars.model = ContinuousModel(
      continuous_blocks_T=continuous_blocks_T, **_vars.__dict__,
    )#.to(device)
    print(' -> Done.\n')

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
  # import sys; sys.path.append('model_architectures/transformer/model_utils')
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

  _vars.optimizer = initialize_optimizer(**_vars.__dict__)
  _vars.criterion = nn.CrossEntropyLoss(ignore_index=_vars.pad_token_id)

  print(f'3. Training models')

  _vars.data_loader_iterators = dict(zip(
    _vars.splits, [iter(_vars.data_loaders[split]) for split in _vars.splits],
  ))

  def get_batch(split):
    batch = next(_vars.data_loader_iterators[split], None)

    if batch is None: 
      _vars.data_loader_iterators[split] = iter(_vars.data_loaders[split])
      batch = next(_vars.data_loader_iterators[split], None)
      if batch is None: 
        raise Exception(f'Length of {split} data loader is 0.')

    input, target = batch['input_ids'], batch['labels']
    batch = (input, target)

    # print(f'input {input.ravel()[:3]} target {target.ravel()[:3]}')

    return batch

  num_epochs_list    = [  int(num_epochs   ) for num_epochs    in _vars.num_epochs   .split('_')]
  levels_list        = [  int(level        ) for level         in _vars.levels_scheme.split('_')]
  learning_rate_list = [float(learning_rate) for learning_rate in _vars.learning_rate.split('_')]
  momentum_list      = [float(momentum     ) for momentum      in _vars.momentum     .split('_')] \
                       if _vars.momentum is not None else [None]*len(levels_list)

  print(f' Starting at level {levels_list[0]}')

  num_training_batches = _vars.num_training_batches \
    if _vars.num_training_batches is not None \
    else len(_vars.data_loaders['training'])
  num_validation_batches = _vars.num_validation_batches \
    if _vars.num_validation_batches is not None \
    else len(_vars.data_loaders['validation'])

  batch = get_batch('training')
  src, tgt = batch
  src = torch.rand(src.shape[0], src.shape[1], _vars.model_dimension).to(_vars.device)
  tgt = torch.rand(tgt.shape[0], tgt.shape[1]-1, _vars.model_dimension).to(_vars.device)
  src.requires_grad_()
  tgt.requires_grad_()
  model_inputs = {
    'input': src, 'target': tgt, 'criterion': nn.MSELoss(),#_vars.criterion, 
    'compute_accuracy': False, 'level': 0,
  }
  loss = _vars.model(**model_inputs)['loss']
  loss.backward()
  ## (I) --> very very similar :) but very large numbers... :( (e+12)
  # print('src.grad', src.grad)
  # print('tgt.grad', tgt.grad)
  # import sys; sys.exit()

  ## (II) --> decoder: very similar. encoder: mostly exact but sometimes quite 
  ## ... different.
  for parameter in _vars.model.parameters(): 
    if parameter.grad is None: print('None')
    else:                      print(parameter.grad.ravel()[:10])
  import sys; sys.exit()

if __name__ == '__main__': main()




