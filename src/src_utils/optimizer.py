import torch

def initialize_optimizer(optimizer_name, model, **kwargs):
  if optimizer_name == 'SGD':
    optimizer = torch.optim.SGD(
      model.parameters(), lr=0., momentum=0.,
    )

  elif optimizer_name == 'Adam':
    optimizer = torch.optim.Adam(
      model.parameters(), lr=0.,
    )

  elif optimizer_name == 'AdamW':
    optimizer = torch.optim.AdamW(
      model.parameters(), lr=0.,
    )

  else: raise Exception()

  return optimizer




