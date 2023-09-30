import torch
import torch.nn as nn

from model.mlp import MLP
from model.multihead_attention import MultiHeadAttention
from model.self_attention import SelfAttention

class TransformerDecoderLayer(nn.Module):
  def __init__(self, _vars):
    super().__init__()
    self.self_attn = SelfAttention(_vars)
    self.cross_attn = MultiHeadAttention(_vars)
    self.mlp = MLP(_vars)
    self.self_attn_layer_norm = nn.LayerNorm(
      (_vars.d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.cross_attn_layer_norm = nn.LayerNorm(
      (_vars.d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.final_layer_norm = nn.LayerNorm(
      (_vars.d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.norm_first = _vars.norm_first

  def forward(self, x, memory, mask_pad_tgt=None, mask_pad_mem=None):
    if self.norm_first:
      x = x + self.self_attn(
        self.self_attn_layer_norm(x), 
        mask_pad=mask_pad_tgt, 
        add_mask_attn=True,
      )
      x = x + self.cross_attn(
        _K=memory, 
        _V=memory, 
        _Q=self.cross_attn_layer_norm(x),
        mask_attn=None,
        mask_pad=mask_pad_mem,
      )
      x = x + self.mlp(self.final_layer_norm(x))

    else: 
      x = self.self_attn_layer_norm(
        x + self.self_attn(
          x, 
          mask_pad=mask_pad_tgt, 
          add_mask_attn=True,
        )
      )
      x = self.cross_attn_layer_norm(
        x + self.cross_attn(
          _K=memory, 
          _V=memory, 
          _Q=x,
          mask_attn=None,
          mask_pad=mask_pad_mem,
        )
      )
      x = self.final_layer_norm(x + self.mlp(x))

    return x




















