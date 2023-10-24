import importlib
import numpy as np
import torch
import torch.nn as nn

from .save import save_model

class ContinuousBlock(nn.Module):
  def __init__(self, ResidualLayer, N, **kwargs):
    super().__init__()
    self.N = N
    self.layers = nn.ModuleList(
      [ContinuousLayer(ResidualLayer=ResidualLayer, seed=0,#i, 
                                                  **kwargs) \
       for i in range(self.N)]
    )
    # self.residual_layers = nn.ModuleList(
    #   [layer.residual_layer for layer in self.layers]
    # )

  def forward(self, **state):
    for i, layer in enumerate(self.layers): state.update(layer(**state))
    return state

class ContinuousLayer(nn.Module):
  def __init__(self, ResidualLayer, seed=None, **kwargs):
    super().__init__()
    if seed is not None: torch.manual_seed(seed)

    self.residual_layer = ResidualLayer(**kwargs)

  def forward(self, x, **kwargs):
    return {'x': x + self.residual_layer(x, **kwargs)['x']}

##
# Transformer encoder layer using their code's scheme & <i>MultiheadAttention</i>
class Model(nn.Module):
  def __init__(self, model_architecture_path, N, 
               seed_precontinuous_block=None, seed_postcontinuous_block=None, 
               **kwargs):
    super().__init__()
    architecture_module = importlib.import_module(model_architecture_path)

    # if seed_precontinuous_block is not None: 
    #   torch.manual_seed(seed_precontinuous_block)
    torch.manual_seed(0)
    self.precontinuous_block = architecture_module.PreContinuousBlock(
      **kwargs
    )
    self.continuous_block = ContinuousBlock(
      ResidualLayer=architecture_module.ContinuousResidualLayer,
      N=N,
      **kwargs,
    )
    # if seed_postcontinuous_block is not None:
    #   torch.manual_seed(seed_postcontinuous_block)
    torch.manual_seed(0)
    self.postcontinuous_block = architecture_module.PostContinuousBlock(
      **kwargs
    )
    ## Continuous block
    # if init_method.lower() != 'none':
    #   print('initializing parameters')
    #   self.init_params()

  def forward(self, **state): 
    return self.static_forward(self, **state)

  @staticmethod
  def static_forward(
    model, input, target=None, criterion=None, compute_accuracy=False, **state,
  ):
    if target is not None or criterion is not None:
      assert target is not None and criterion is not None
    if compute_accuracy: 
      assert target is not None and isinstance(criterion, nn.CrossEntropyLoss)

    state['x'] = input
    state.update(model.precontinuous_block (**state))
    state.update(model.continuous_block    (**state))
    state.update(model.postcontinuous_block(**state))

    if target is not None:
      if isinstance(criterion, nn.CrossEntropyLoss):
        logits = state['x']
        loss = criterion(
          logits.view(-1, logits.shape[-1]), 
          target.view(-1),
        )
        state['logits'] = logits

        if compute_accuracy:
          with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            pad_id = criterion.__dict__['ignore_index']
            not_pad = (target != pad_id)
            correct = ((predictions == target) * not_pad).sum().item()
            total = not_pad.sum().item()

            state['predictions'] = predictions
            state['correct'] = correct
            state['total'] = total

      elif isinstance(criterion, nn.MSELoss):
        output = state['x']
        loss = criterion(logits, target)

        state['output'] = output

      else: raise Exception()

      state['loss'] = loss

    return state

  def save(**kwargs): 
    '''Arguments: fn_without_extension=None, models_dir=None, optimizer=None, 
                  **other''' 
    self.static_save(self, **kwargs)

  @staticmethod
  def static_save(model, **kwargs):
    save_model(model, **kwargs)

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



