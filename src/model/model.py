import importlib
import numpy as np
import torch
import torch.nn as nn

class ContinuousBlock(nn.Module):
  def __init__(self, ResidualLayer, N):
    super().__init__()
    self.N = N
    self.layers = nn.ModuleList(
      [ContinuousLayer(ResidualLayer=ResidualLayer, seed=i) \
       for i in range(self.N)]
    )
    self.residual_layers = nn.ModuleList(
      [layer.residual_layer for layer in self.layers]
    )

  def forward(self, **state):
    for i, layer in enumerate(self.layers): state.update(layer(**state))
    return state

class ContinuousLayer(nn.Module):
  def __init__(self, ResidualLayer, seed=None):
    super().__init__()
    self.residual_layer = ResidualLayer(seed=seed)

  def forward(self, x, **kwargs):
    return {'x': x + self.residual_layer(x, **kwargs)['x']}

##
# Transformer encoder layer using their code's scheme & <i>MultiheadAttention</i>
class Model(nn.Module):
  def __init__(self, model_architecture_path, N):
    super().__init__()
    architecture_module = importlib.import_module(model_architecture_path)

    self.precontinuous_block  = architecture_module.PreContinuousBlock()
    self.postcontinuous_block = architecture_module.PostContinuousBlock()
    self.continuous_block     = ContinuousBlock(
      ResidualLayer=architecture_module.ContinuousResidualLayer,
      N=N,
    )

    ## Continuous block
    # if init_method.lower() != 'none':
    #   print('initializing parameters')
    #   self.init_params()

  def forward(self, x):
    state = {'x': x}
    state.update(self.precontinuous_block (**state))
    state.update(self.continuous_block    (**state))
    state.update(self.postcontinuous_block(**state))
    return state

  # def init_params(self):
  #   self.apply(self._init_layers)

  # def _init_layers(self, m):
  #   classname = m.__class__.__name__
  #   if isinstance(m, nn.Conv2d):
  #     if m.weight is not None:
  #       torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

  #     if m.bias is not None:
  #       torch.nn.init.constant_(m.bias, 0)
    
  #   if isinstance(m, nn.BatchNorm2d):
  #     if m.weight is not None:
  #       torch.nn.init.constant_(m.weight, 1)

  #     if m.bias is not None:
  #       torch.nn.init.constant_(m.bias, 0)
    
  #   if isinstance(m, nn.Linear):
  #     if m.weight is not None:
  #       torch.nn.init.normal_(m.weight)

  #     if m.bias is not None:
  #       torch.nn.init.constant_(m.bias, 0)
    
  #   if isinstance(m, nn.MultiheadAttention):
  #     print(f'Init method: {self.init_method}')
  #     if self.init_method == 'Normal':
  #       m.in_proj_weight.data.normal_(mean=0.0, std=0.02)
  #       m.out_proj.weight.data.normal_(mean=0.0, std=0.02)
  #     elif self.init_method == 'Xavier':
  #       torch.nn.init.xavier_uniform_(m.in_proj_weight, gain=np.sqrt(2))
  #       torch.nn.init.xavier_uniform_(m.out_proj.weight, gain=np.sqrt(2))
  #     else:
  #       Exception('Initialization method unknown')



