import torch.nn as nn

from ..transformer_utils.F_dec import F_dec

class ContinuousResidualLayer(nn.Module):
  def __init__(self, model_dimension, num_heads, dim_ff, **kwargs):
    super().__init__()

    self.F = F_dec(model_dimension, num_heads, dim_ff)

    # self.apply(init_weights)

  def forward(self, x, y, mask_pad_tgt, mask_pad_mem, **kwargs):  # x: [b, L , d]
                                                                  # y: [b, L', d]
    y = self.F(
      x=y, memory=x, mask_pad_tgt=mask_pad_tgt, mask_pad_mem=mask_pad_mem
    )

    return {'y': y}



