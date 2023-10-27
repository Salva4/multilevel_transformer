## Taken from Karpathy's github: [url]

import torch

# data loading
def get_batch(
  split, train_data, val_data, context_window, batch_size, device
):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - context_window, (batch_size,))
  x = torch.stack([data[i : i + context_window] for i in ix])
  y = torch.stack([data[i+1 : i+1 + context_window] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss(
  model, eval_iters, train_data, val_data, device, criterion, context_window, 
  batch_size, mgrit, mgopt, relaxation, num_iterations, **kwargs,
):
  out = {}
  model.eval()

  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
      torch.manual_seed(-(k+1))
      batch = get_batch(
        split, train_data, val_data, context_window, batch_size, device
      )
      input_ids, target_ids = batch
      model_inputs = {
        'input': input_ids, 'target': target_ids, 'criterion': criterion, 
        'compute_accuracy': False, 'use_MGRIT': mgrit, 'use_MGOPT': mgopt, 
        'relaxation': relaxation, 'num_iterations': num_iterations,
      }
      model_outputs = model(**model_inputs)
      loss = model_outputs['loss']
      losses[k] = loss.item()

    out[split] = losses.mean()

  model.train()

  return out

def train_batch(
  model, train_data, val_data, device, optimizer, criterion, context_window, 
  batch_size, mgrit, mgopt, relaxation, num_iterations, **kwargs,
):
  batch = get_batch(
    'train', train_data, val_data, context_window, batch_size, device,
  )
  input_ids, target_ids = batch
  model_inputs = {
    'input': input_ids, 'target': target_ids, 'criterion': criterion, 
    'compute_accuracy': False, 'use_MGRIT': mgrit, 'use_MGOPT': mgopt, 
    'relaxation': relaxation, 'num_iterations': num_iterations,
  }
  model_outputs = model(**model_inputs)#, **kwargs)
  loss = model_outputs['loss']
  print(f'loss {loss.item()}')

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


