import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, x, max_new_tokens, context_window, **fwd_pass_details):
  for _ in range(max_new_tokens):
    x_cond = x[:, -context_window:]
    model_inputs = {'input': x_cond}
    model_inputs.update(fwd_pass_details)
    model_outputs = model(**model_inputs)#, **kwargs)
    logits = model_outputs['x']
    logits = logits[:, -1, :] # becomes (B, C)
    probs = F.softmax(logits, dim=-1) # (B, C)
    x_next = torch.multinomial(probs, num_samples=1)#.to(x.device) # (B, 1)
    x = torch.cat((x, x_next), dim=1) # (B, T+1)

  return x