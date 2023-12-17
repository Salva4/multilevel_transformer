import numpy as np
import os
import torch
import torch.nn as nn
import sys

from .continuous_block         import ContinuousBlock
from .continuous_block_enc_dec import ContinuousBlock \
                               as ContinuousBlock_encoder_decoder_transformer

sys.path.append(os.path.join('..'))
from src_utils.filter_dict import filter_keys

class ContinuousModel(nn.Module):
  def __init__(
    self, model, continuous_blocks_T, is_encoder_decoder_transformer=False,
    **kwargs_continuous_block,
  ):
    super().__init__()
    self.model = model
    # self.register_buffer('model', model)
    # self.interpol = kwargs['interpol']

    ## Exceptional case regarding GradientFunction: encoder-decoder
    ## transformer.
    ## This code handles any model where the continuous blocks are computed
    ## sequentially (linear model in example1, encoder-only transformer,
    ## decoder-only transformer). Exceptionally, it also handles an
    ## encoder-decoder transformer architecture, where the memory appears
    ## several times in the computation of the target embeddings. For this
    ## case, a different ContinuousBlock (with a different GradientFunction
    ## for the decoder) is used.
    if is_encoder_decoder_transformer:
      ContinuousBlock_module = ContinuousBlock_encoder_decoder_transformer
    else:
      ContinuousBlock_module = ContinuousBlock

    self.precontinuous_block  = self.model.precontinuous_block
    self.continuous_blocks = nn.ModuleList(
      [
        ContinuousBlock_module(
          ψ=nn.ModuleList(
            [layer.residual_layer for layer in continuous_block.layers]
          ),
          N=continuous_block.num_layers,
          T=continuous_blocks_T[continuous_block_idx],
          state_symbol=continuous_block.state_symbol,
          **filter_keys(kwargs_continuous_block, ('T',)),
        ) \
        for continuous_block_idx, continuous_block in enumerate(
          self.model.continuous_blocks
        )
      ]
    )
    self.postcontinuous_block = self.model.postcontinuous_block

    # if self.init_method != 'None':
    #   self.init_params()

    if hasattr(self.model, 'device'):
      self.device = self.model.device
      self.to(self.device)

  def forward(self, **state): return self.model.static_forward(self, **state)

  def interpolate_weights(self, fine_level, interpolation):
    for continuous_block in self.continuous_blocks:
      continuous_block.interpolate_weights(fine_level, interpolation)

  def train_(self, **kwargs):
    '''Arguments:
      optimizer, device, criterion, get_batch, num_batches,
      compute_accuracy=False, print_times=False, use_mgopt=False, **details,
    '''
    return self.model.static_train(self, **kwargs)

  def evaluate(self, **kwargs):
    '''Arguments:
      device, criterion, get_batch, num_batches, compute_accuracy=False,
      print_times=False, **fwd_pass_details,
    '''
    return self.model.static_evaluate(self, **kwargs)

  def save(self, **kwargs):
    '''Arguments:
      fn_without_extension=None, models_dir=None, optimizer=None, **other
    '''
    self.model.static_save(self, **kwargs)

  def load(self, **kwargs):
    '''Arguments: model, model_path, optimizer=None'''
    return self.model.static_load(self, **kwargs)

  # def init_params(self):
  #   self.apply(self._init_layers)

  # def _init_layers(self, m):
  #   classname = m.__class__.__name__
  #   if isinstance(m, nn.Conv2d):
  #     if m.weight is not None:
  #       torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

  #     if m.bias is not None:
  #       torch.nn.init.constant_(m.bias, 0)

  #   if isinstance(m, nn.BatchNorm2d):
  #     if m.weight is not None:
  #       torch.nn.init.constant_(m.weight, 1)

  #     if m.bias is not None:
  #       torch.nn.init.constant_(m.bias, 0)

  #   if isinstance(m, nn.Linear):
  #     if m.weight is not None:
  #       torch.nn.init.normal_(m.weight)

  #     if m.bias is not None:
  #       torch.nn.init.constant_(m.bias, 0)

  #   if isinstance(m, nn.MultiheadAttention):
  #     if self.init_method == 'Normal':
  #       m.in_proj_weight.data.normal_(mean=0.0, std=0.02)
  #       m.out_proj.weight.data.normal_(mean=0.0, std=0.02)
  #     elif self.init_method == 'Xavier':
  #       torch.nn.init.xavier_uniform_(m.in_proj_weight, gain=np.sqrt(2))
  #       torch.nn.init.xavier_uniform_(m.out_proj.weight, gain=np.sqrt(2))
  #     else:
  #       Exception('Initialization method unknown')




