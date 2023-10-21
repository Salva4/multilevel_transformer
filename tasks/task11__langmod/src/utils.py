## Taken from Karpathy's github: [url]

import torch

# data loading
def get_batch(split, train_data, val_data, block_size, batch_size, device):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss(
    model, eval_iters, train_data, val_data, max_len, batch_size, device, 
    criterion,
):
  out = {}
  model.eval()

  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
      batch = get_batch(
        split, train_data, val_data, max_len, batch_size, device
      )
      input_ids, target_ids = batch
      model_inputs = {
        'input': input_ids, 'target': target_ids, 'criterion': criterion, 
        'compute_accuracy': False,
      }
      model_outputs = model(**model_inputs)
      loss = model_outputs['loss']
      losses[k] = loss.item()

    out[split] = losses.mean()

  model.train()

  return out


