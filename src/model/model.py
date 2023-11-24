import copy
import importlib
import numpy as np
import torch
import torch.nn as nn
import sys

from .save import save_model, load_model
from .train import train, evaluate

# from ..src_utils.filter_dict import filter_keys
sys.path.append('..')
from src_utils.filter_dict import filter_keys

class ContinuousBlock(nn.Module):
  def __init__(
    self, ResidualLayer, num_layers, continuous_block_idx, **kwargs,
  ):
    super().__init__()
    self.num_layers = num_layers
    self.layers = nn.ModuleList(
      [ContinuousLayer(ResidualLayer=ResidualLayer, #seed=0,#i,
                                                  **kwargs) \
       for i in range(self.num_layers)]
    )
    self.continuous_block_idx = continuous_block_idx
    self.name = ResidualLayer.name if 'name' in dir(ResidualLayer) \
           else f'continuous_block_{continuous_block_idx}'
    self.state_symbol = self.layers[0].state_symbol

  def forward(self, store_hidden_states, **state):
    for i, layer in enumerate(self.layers):
      state.update(layer(**state))

      if store_hidden_states:
        if not self.name in state['hidden_states']: 
          state['hidden_states'][self.name] = []

        state['hidden_states'][self.name].append(
          state[self.state_symbol].clone(),
        )

    return state

class ContinuousLayer(nn.Module):
  def __init__(self, ResidualLayer, seed=None, **kwargs):
    super().__init__()
    # if seed is not None: torch.manual_seed(seed)

    self.residual_layer = ResidualLayer(**kwargs)
    self.state_symbol = self.residual_layer.state_symbol \
      if 'state_symbol' in dir(ResidualLayer) else 'x'


  def forward(self, **kwargs):
    state, state_symbol = kwargs[self.state_symbol], self.state_symbol
    return {
      state_symbol: state + self.residual_layer(**kwargs)[state_symbol],
    }

##
# Transformer encoder layer using their code's scheme & <i>MultiheadAttention</i>
class Model(nn.Module):
  def __init__(
    self, model_name, continuous_blocks_num_layers, initialize_weights=False,
    **kwargs,
  ):
               # seed_precontinuous_block=None, seed_postcontinuous_block=None,
    super().__init__()
    kwargs['model'] = self

    model_architecture_path = '.'.join(
      ['model_architectures', model_name, 'architecture']
    )

    ## Pre-continuous block
    precontinuous_block_module = importlib.import_module(
      '.'.join([model_architecture_path, 'precontinuous_block'])
    )
    # if seed_precontinuous_block is not None:
    #   torch.manual_seed(seed_precontinuous_block)
    # torch.manual_seed(0)
    self.precontinuous_block = \
      precontinuous_block_module.PreContinuousBlock(**kwargs)

    ## Continuous block
    num_continuous_blocks = len(continuous_blocks_num_layers)
    self.continuous_blocks = nn.ModuleList()

    for continuous_block_idx in range(num_continuous_blocks):
      continuous_block_module = importlib.import_module('.'.join(
        [model_architecture_path, f'continuous_block_{continuous_block_idx+1}']
      ))
      num_layers = continuous_blocks_num_layers[continuous_block_idx]
      continuous_block = ContinuousBlock(
        ResidualLayer=continuous_block_module.ContinuousResidualLayer,
        num_layers=num_layers,
        continuous_block_idx=continuous_block_idx,
        **filter_keys(
          kwargs, ('ResidualLayer', 'num_layers', 'continuous_block_idx'),
        ),
      )
      self.continuous_blocks.append(continuous_block)

    ## Post-continuous block
    postcontinuous_block_module = importlib.import_module(
      '.'.join([model_architecture_path, 'postcontinuous_block'])
    )
    # if seed_postcontinuous_block is not None:
    #   torch.manual_seed(seed_postcontinuous_block)
    # torch.manual_seed(0)
    self.postcontinuous_block = \
      postcontinuous_block_module.PostContinuousBlock(**kwargs)
    
    ## Weights initialization
    if initialize_weights:
      # for p in self.parameters(): p.data.zero_()
      print('Initializing weights...')
      self.apply(self.initialize_weights)

    if kwargs.get('device', None) is not None:
      self.device = kwargs['device']
      self.to(self.device)

  def initialize_weights(self, module):    
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0., std=1.)
      if module.bias is not None: module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0., std=1.)

      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()

    # else: print(module)

  def forward(self, **state):
    return self.static_forward(self, **state)

  @staticmethod
  def static_forward(
    model, input, target=None, criterion=None, compute_accuracy=False, 
    store_hidden_states=False, **state,
  ):
    if target is not None or criterion is not None:
      assert target is not None and criterion is not None
    if compute_accuracy:
      assert target is not None and isinstance(criterion, nn.CrossEntropyLoss)

    state['x'] = state['input' ] = input
    state['y'] = state['target'] = target

    state['hidden_states'] = hidden_states = {} if store_hidden_states \
                                                else None

    ## Forward pass ###################
    state.update(model.precontinuous_block (**state))

    if store_hidden_states: 
      hidden_states['precontinuous_block'] = {
        'x': state['x'].clone(), 'y': state['y'].clone(),
      }

    for i, continuous_block in enumerate(model.continuous_blocks):
      state.update(
        continuous_block(**state, store_hidden_states=store_hidden_states)
      )

      # if store_hidden_states: 
      #   hidden_states[f'continuous_block_{i}'].append(
      #     state[continuous_block.state_symbol].clone(),
      #   )

    state.update(model.postcontinuous_block(**state))

    if store_hidden_states:
      hidden_states['postcontinuous_block'] = {
        'x': state['x'].clone(), 'y': state['y'].clone(),
      }
    ###################################
    
    target = state['target']

    # print(f'''state['x'].shape {state['x'].shape}''')
    # print(f'''state['y'].shape {state['y'].shape}''')
    # print(f'''state['target'].shape {state['target'].shape}''')

    if target is not None:
      if isinstance(criterion, nn.CrossEntropyLoss):
        logits = state['x']
        # print('shapes', logits.shape, target.shape)
        loss = criterion(logits.transpose(1,2), target)
        # loss = criterion(
        #   logits.view(-1, logits.shape[-1]),
        #   target.view(-1),
        # )
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

  def train_(self, *args, **kwargs):
    '''Arguments:
      optimizer, device, criterion, get_batch, num_batches,
      compute_accuracy=False, print_times=False, use_mgopt=False (!),
      **details,
    '''
    assert not kwargs.get('use_mgopt', False)
    return self.static_train(self, *args, **kwargs)

  @staticmethod
  def static_train(model, *args, **kwargs):
    return train(model, *args, **kwargs)

  def evaluate(self, *args, **kwargs):
    '''Arguments:
      device, criterion, get_batch, num_batches, compute_accuracy=False,
      print_times=False, **fwd_pass_details,
    '''
    return self.static_evaluate(self, *args, **kwargs)

  @staticmethod
  def static_evaluate(model, *args, **kwargs):
    return evaluate(model, *args, **kwargs)

  def save(self, **kwargs):
    '''Arguments:
      fn_without_extension=None, models_dir=None, optimizer=None, **other
    '''
    self.static_save(self, **kwargs)

  @staticmethod
  def static_save(model, **kwargs): save_model(model, **kwargs)

  def load(self, **kwargs):
    '''Arguments: model, model_path, optimizer=None'''
    return self.static_load(self, **kwargs)

  @staticmethod
  def static_load(model, **kwargs): return load_model(model, **kwargs)

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



