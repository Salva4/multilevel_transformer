import torch

## No-debug:
# @torch.no_grad()
# def solve_sequential(x0, N, T, c, Φ, F, **kwargs):
#   h = T/N
#   x = torch.zeros(
#     size=(N+1, *x0.shape), dtype=torch.float32, device=x0.device,
#   )

#   x[0] = x0

#   for i in range(N):
#     t = i*h
#     x[i+1] = Φ(t=t, x=x[i], h=h, F=F, **kwargs)

#   return x

## Debug 2/2 (1/2 in continuous_block):
def solve_sequential(x0, N, T, c, Φ, F, **kwargs):
  h = T/N
  x = x0

  for i in range(N):
    t = i*h
    x = Φ(t=t, x=x, h=h, F=F, **kwargs)

  return x


