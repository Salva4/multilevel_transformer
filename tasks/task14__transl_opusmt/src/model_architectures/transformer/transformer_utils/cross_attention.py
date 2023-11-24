import torch
import torch.nn as nn

# from .multihead_attention import MultiHeadAttention
import sys; sys.path.append('model_architectures/transformer/transformer_utils')
from multihead_attention import MultiHeadAttention

class CrossAttention(nn.Module):
  def __init__(self, model_dimension, num_heads):
    super().__init__()
    self.attn = MultiHeadAttention(model_dimension, num_heads)
    # self.attn = nn.MultiheadAttention(
    #   embed_dim=model_dimension, num_heads=num_heads, batch_first=True,
    # )

  def forward(self, _K, _V, _Q, mask_pad=None):  # _K: [b, L , d]
    out = self.attn(                             # _V: [b, L , d]
      _K=_K,                                     # _Q: [b, L', d]
      _V=_V,
      _Q=_Q,
      mask_attn=None,
      mask_pad=mask_pad,
    )  # out: [b, L', d]
    # out = self.attn(                             # _V: [b, L , d]
    #   key=_K,                                     # _Q: [b, L', d]
    #   value=_V,
    #   query=_Q,
    #   attn_mask=None,
    #   key_padding_mask=mask_pad,
    # )[0]  # out: [b, L', d]

    return out
