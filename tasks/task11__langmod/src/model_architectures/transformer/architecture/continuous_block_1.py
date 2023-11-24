import torch.nn as nn

# from .methods.init_weights import init_weights
from ..transformer_utils.F_dec import F_dec

class ContinuousResidualLayer(nn.Module):
  ''' 
  Insert here the optional arguments "name" and "state_symbol".
  By default, name='continuous_block_{idx}' and state_symbol='x'.
  '''
  def __init__(self, **kwargs):
    super().__init__()
    self.F = F_dec(**kwargs)
    # self.apply(init_weights)

  def forward(self, x, **kwargs): #return {'x': self.F(x)}
    x = self.F(x)

    return {'x': x}




