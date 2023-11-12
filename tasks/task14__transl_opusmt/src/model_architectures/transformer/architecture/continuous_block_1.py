import torch.nn as nn

from ..transformer_utils.F_enc import F_enc

class ContinuousResidualLayer(nn.Module):
  def __init__(self, model_dimension, num_heads, dim_ff, **kwargs):
    super().__init__()

    self.F = F_enc(model_dimension, num_heads, dim_ff)

    # self.apply(init_weights)

  def forward(self, x, mask_pad_src, **kwargs):  # x: [b, L, d]
    x = self.F(
      x=x, mask_pad_src=mask_pad_src,
    )

    return {'x': x}




