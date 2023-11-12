import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttention

class SelfAttention(nn.Module):
  def __init__(self, model_dimension, num_heads):
    super().__init__()
    self.attn = MultiHeadAttention(model_dimension, num_heads)

  def forward(self, x, mask_pad=None, add_mask_attn=False):  # x: [b, L, d]
    L = x.shape[1]

    mask_attn = nn.Transformer.generate_square_subsequent_mask(L).to(x.device) \
                if add_mask_attn else None

    out = self.attn(
      _K=x, 
      _V=x, 
      _Q=x,
      mask_attn=mask_attn,
      mask_pad=mask_pad,
    )  # out: [b, L, d]
    
    return out
