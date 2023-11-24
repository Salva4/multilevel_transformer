import torch.nn as nn

# from .self_attention import SelfAttention
# from .cross_attention import CrossAttention
# from .mlp import MLP
import sys; sys.path.append('model_architectures/transformer/transformer_utils')
from self_attention import SelfAttention
from cross_attention import CrossAttention
from mlp import MLP

class F_dec(nn.Module):
  def __init__(self, model_dimension, num_heads, dim_ff):
    super().__init__()
    self.model_dimension = model_dimension
    
    self.self_attn = SelfAttention(model_dimension, num_heads)
    self.cross_attn = CrossAttention(model_dimension, num_heads)
    self.mlp = MLP(model_dimension, dim_ff)
    self.self_attn_layer_norm = nn.LayerNorm(
      (self.model_dimension,), 
      # eps=1e-5,
      # elementwise_affine=True,
    )
    self.cross_attn_layer_norm = nn.LayerNorm(
      (self.model_dimension,), 
      # eps=1e-5,
      # elementwise_affine=True,
    )
    self.mlp_layer_norm = nn.LayerNorm(
      (self.model_dimension,), 
      # eps=1e-5,
      # elementwise_affine=True,
    )
    self.phi1 = lambda x, mask_pad_tgt: self.self_attn(
      x=self.self_attn_layer_norm(x),
      mask_pad=mask_pad_tgt,
      add_mask_attn=True,
    )
    self.phi2 = lambda x: self.mlp(self.mlp_layer_norm(x))
    self.phi3 = lambda x, memory, mask_pad_mem: self.cross_attn(
      _K=memory, 
      _V=memory, 
      _Q=self.cross_attn_layer_norm(x),
      mask_pad=mask_pad_mem,
    )

  def forward(self, x, memory, mask_pad_tgt=None, mask_pad_mem=None):
    phi1_x = self.phi1(x, mask_pad_tgt)
    phi3_x = self.phi3(x + phi1_x, memory, mask_pad_mem)
    phi2_x = self.phi2(x + phi1_x + phi3_x)
    return phi1_x + phi3_x + phi2_x

    # x0 = x
    # x1 = self.self_attn(
    #   x=self.self_attn_layer_norm(x0.clone()),
    #   mask_pad=mask_pad_tgt.clone(),
    #   add_mask_attn=True,
    # )
    # x2 = self.cross_attn(
    #   _K=memory.clone(), 
    #   _V=memory.clone(), 
    #   _Q=self.cross_attn_layer_norm(x0.clone() + x1.clone()),
    #   mask_pad=mask_pad_mem.clone(),
    # )
    # x3 = self.mlp(self.mlp_layer_norm(x0.clone() + x1.clone() + x2.clone()))
    # return x1.clone() + x2.clone() + x3.clone()

    # x = x + self.self_attn(
    #   x=self.self_attn_layer_norm(x),
    #   mask_pad=mask_pad_tgt,
    #   add_mask_attn=False,#True,
    # )
    # x = x + self.cross_attn(
    #   _K=memory,
    #   _V=memory,
    #   _Q=self.cross_attn_layer_norm(x),
    #   mask_pad=mask_pad_mem,
    # )
    # x = x + self.mlp(self.mlp_layer_norm(x))
    # return x




