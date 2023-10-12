import torch
import torch.nn as nn

from model.mlp import MLP
from model.self_attention import SelfAttention

class TransformerEncoderLayer(nn.Module):
  def __init__(self, _vars):
    super().__init__()
    self.self_attn = SelfAttention(_vars)
    self.mlp = MLP(_vars)
    self.self_attn_layer_norm = nn.LayerNorm(
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

  def forward(self, x, mask_pad_src=None):
    if self.norm_first:
      x = x + self.self_attn(
          x=self.self_attn_layer_norm(x),
          mask_pad=mask_pad_src,
          add_mask_attn=False,
        )
      x = x + self.mlp(self.final_layer_norm(x))

    else: 
      x = self.self_attn_layer_norm(
        x + self.self_attn(
          x=x,
          mask_pad=mask_pad_src,
          add_mask_attn=False,
        )
      )
      x = self.final_layer_norm(x + self.mlp(x))

    return x






















