import numpy as np
import torch
import inspect

def get_batch(_vars):
  batch = torch.zeros((_vars.batch_size, _vars.chunk_size + 1), 
                                              dtype=torch.long)
  ids = np.random.randint(len(_vars.data) - _vars.chunk_size - 1, 
                                         size=(_vars.batch_size))
  for j, id in enumerate(ids):
    batch[j, :] = _vars.data[id : id + _vars.chunk_size + 1]
  return batch.to(_vars.dev)

def pass_args(f, args, **kwargs): 
  keys = filter(lambda x: x in args, inspect.signature(f).parameters)
  args_f = {k: args.__dict__[k] for k in keys}
  return f(**args_f, **kwargs)
