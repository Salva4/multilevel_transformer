## Taken from Karpathy's github: [url]

import time
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
  model, eval_iters, train_data, val_data, device, criterion, batch_size, 
  context_window, **fwd_pass_details,
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
        'compute_accuracy': False,
      }
      model_inputs.update(fwd_pass_details)
      model_outputs = model(**model_inputs)
      loss = model_outputs['loss']
      losses[k] = loss.item()

    out[split] = losses.mean()

  model.train()

  return out

def train_batch(
  model, train_data, val_data, device, optimizer, criterion, batch_size, 
  context_window, **fwd_pass_details,
):
  get_batch_time_start = time.time()
  batch = get_batch(
    'train', train_data, val_data, context_window, batch_size, device,
  )
  get_batch_time_end = time.time()

  input_ids, target_ids = batch
  model_inputs = {
    'input': input_ids, 'target': target_ids, 'criterion': criterion, 
    'compute_accuracy': False,
  }
  model_inputs.update(fwd_pass_details)

  batch_fwd_time_start = time.time()
  model_outputs = model(**model_inputs)
  batch_fwd_time_end = time.time()

  loss = model_outputs['loss']
  # print(f'loss {loss.item()}')

  optimizer.zero_grad(set_to_none=True)

  batch_bwd_time_start = time.time()
  loss.backward()
  batch_bwd_time_end = time.time()

  optimizer.step()

  print(loss.item())

  # print(f'Getting batch time: {get_batch_time_end - get_batch_time_start} seconds')
  # print(f'Training batch fwd pass time: {batch_fwd_time_end - batch_fwd_time_start} seconds')
  # print(f'Training batch bwd pass time: {batch_bwd_time_end - batch_bwd_time_start} seconds')

  # print(next(model.precontinuous_block.parameters()).ravel()[:5])
  # print(next(model.continuous_block.parameters()).ravel()[:5])
  # print(next(model.postcontinuous_block.parameters()).ravel()[:5])








