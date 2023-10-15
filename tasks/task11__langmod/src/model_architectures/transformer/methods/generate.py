import torch
import torch.nn.functional as F

def generate(model, x, max_new_tokens, block_size, **kwargs):
  for _ in range(max_new_tokens):
    x_cond = x[:, -block_size:]
    model_inputs = {'x': x_cond}
    outputs = model(**model_inputs)
    logits = outputs['x']
    logits = logits[:, -1, :] # becomes (B, C)
    probs = F.softmax(logits, dim=-1) # (B, C)
    x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    x = torch.cat((x, x_next), dim=1) # (B, T+1)
  return x