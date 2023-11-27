import torch
import torch.nn as nn

# from .multihead_attention import MultiHeadAttention
import sys; sys.path.append('model_architectures/transformer/model_utils')
from multihead_attention import MultiHeadAttention

class SelfAttention(nn.Module):
  def __init__(self, model_dimension, num_heads):
    super().__init__()
    self.attn = MultiHeadAttention(model_dimension, num_heads)
    # self.attn = nn.MultiheadAttention(
    #   embed_dim=model_dimension, num_heads=num_heads, batch_first=True,
    # )

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
    # out = self.attn(
    #   key=x, 
    #   value=x, 
    #   query=x,
    #   attn_mask=mask_attn,
    #   key_padding_mask=mask_pad,
    # )[0]  # out: [b, L, d]
    
    return out
