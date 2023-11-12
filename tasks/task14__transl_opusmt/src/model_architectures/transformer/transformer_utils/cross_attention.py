import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttention

class CrossAttention(nn.Module):
  def __init__(self, model_dimension, num_heads):
    super().__init__()
    self.attn = MultiHeadAttention(model_dimension, num_heads)

  def forward(self, _K, _V, _Q, mask_pad=None):  # _K: [b, L , d]
                                                 # _V: [b, L , d]
                                                 # _Q: [b, L', d]
    out = self.attn(
      _K=_K, 
      _V=_V, 
      _Q=_Q,
      mask_attn=None,
      mask_pad=mask_pad,
    )  # out: [b, L', d]

    return out
