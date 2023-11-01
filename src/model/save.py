import datetime as dt
import os
import torch

def find_models_dir():
  outputs_dir_items = os.listdir(os.path.join('..', 'outputs'))
  candidates = [
    fn for fn in outputs_dir_items if fn.startswith('continuous_transformer')
  ]
  if len(candidates) == 0: return '.'
  models_dir = os.path.join('..', 'outputs', candidates[-1], 'models')
  if not os.path.exists(models_dir): os.mkdir(models_dir)
  return models_dir

def load_model(model, model_name, models_dir=None, optimizer=None):
  model_name += '.pt'
  if models_dir is None: models_dir = find_models_dir()
  model_path = os.path.join(models_dir, model_name)

  if not os.path.exists(model_path): 
    # print('The model could not be loaded because the path does not exist.')
    return {'error': 'The path does not exist.'}
  
  model_state = torch.load(model_path)

  model.load_state_dict(model_state.pop('model_state'))

  if optimizer is not None:
    if 'optimizer_state' in model_state:
      optimizer.load_state_dict(model_state.pop('optimizer_state'))
    else: print('No saved optimizer_state found.')

  return model_state

def save_model(model, fn_without_extension=None, models_dir=None, 
               optimizer=None, **other):
  fn = (
    fn_without_extension if fn_without_extension is not None else \
    dt.datetime.now().strftime('%Y%m%d%H%M%S')
  ) + '.pt'
  if models_dir is None: models_dir = find_models_dir()
  model_path = os.path.join(models_dir, fn)

  model_state = {}
  model_state['model_state'] = model.state_dict()

  if optimizer is not None: 
    model_state['optimizer_state'] = optimizer.state_dict()

  model_state.update(other)
  torch.save(model_state, model_path)














