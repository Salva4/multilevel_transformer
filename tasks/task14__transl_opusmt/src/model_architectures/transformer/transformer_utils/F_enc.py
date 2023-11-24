import torch.nn as nn

# from .self_attention import SelfAttention
# from .mlp import MLP
import sys; sys.path.append('model_architectures/transformer/transformer_utils')
from self_attention import SelfAttention
from mlp import MLP

class F_enc(nn.Module):
  def __init__(self, model_dimension, num_heads, dim_ff):
    super().__init__()
    self.model_dimension = model_dimension

    self.self_attn = SelfAttention(model_dimension, num_heads)
    self.mlp = MLP(model_dimension, dim_ff)
    self.self_attn_layer_norm = nn.LayerNorm(
      (self.model_dimension,), 
      # eps=1e-5, 
      # elementwise_affine=True,
    )
    self.mlp_layer_norm = nn.LayerNorm(
      (self.model_dimension,), 
      # eps=1e-5, 
      # elementwise_affine=True,
    )
    self.phi1 = lambda x, mask_pad_src: self.self_attn(
      x=self.self_attn_layer_norm(x),
      mask_pad=mask_pad_src,
      add_mask_attn=False,
    )
    self.phi2 = lambda x: self.mlp(self.mlp_layer_norm(x))

  def forward(self, x, mask_pad_src=None):
    phi1_x = self.phi1(x, mask_pad_src)
    phi2_x = self.phi2(x + phi1_x)
    return phi1_x + phi2_x

    # x0 = x
    # x1 = self.self_attn(
    #   x=self.self_attn_layer_norm(x0.clone()),
    #   mask_pad=mask_pad_src.clone(),
    #   add_mask_attn=False,
    # )
    # x2 = self.mlp(self.mlp_layer_norm(x0.clone() + x1.clone()))
    # return x1.clone() + x2.clone()

    # x = x + self.self_attn(
    #   x=self.self_attn_layer_norm(x),
    #   mask_pad=mask_pad_src,
    #   add_mask_attn=False,
    # )
    # x = x + self.mlp(self.mlp_layer_norm(x))
    # return x




