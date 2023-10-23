import datetime as dt
import os
import torch

def find_models_dir():
  outputs_dir_items = os.listdir(os.path.join('..', 'outputs'))
  candidates = [
    fn for fn in outputs_dir_items if fn.startswith('continuous_transformer')
  ]
  if len(candidates)==0 or len(candidates)>1: return False
  models_dir = os.path.join('..', 'outputs', candidates[0], 'models')
  return models_dir

def save_model(model, fn_without_extension=None, models_dir=None, 
               optimizer=None, **other):
  fn = (
    fn_without_extension if fn_without_extension is not None else \
    dt.datetime.now().strftime('%Y%m%d%H%M%S')
  ) + '.pt'
  if models_dir is None: models_dir = find_models_dir()
  path = os.path.join(models_dir, fn)

  model_state = {}
  model_state['model_state'] = model.state_dict()

  if optimizer is not None: 
    model_state['optimizer_state'] = optimizer.state_dict()

  model_state.update(other)
  torch.save(model_state, path)














