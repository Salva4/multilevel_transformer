import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(
  model, x, max_new_tokens, context_window, mgrit, mgopt, relaxation, 
  num_iterations, **kwargs,
):
  for _ in range(max_new_tokens):
    x_cond = x[:, -context_window:]
    model_inputs = {
      'input': x_cond, 'use_MGRIT': mgrit, 'use_MGOPT': mgopt, 
      'relaxation': relaxation, 'num_iterations': num_iterations,
    }
    model_outputs = model(**model_inputs)#, **kwargs)
    logits = model_outputs['x']
    logits = logits[:, -1, :] # becomes (B, C)
    probs = F.softmax(logits, dim=-1) # (B, C)
    x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    x = torch.cat((x, x_next), dim=1) # (B, T+1)

  return x