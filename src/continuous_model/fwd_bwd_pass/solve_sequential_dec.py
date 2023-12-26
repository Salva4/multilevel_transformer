import torch

## No-debug:
# @torch.no_grad()
# def solve_sequential(x, y, N, T, c, Φ, F, **kwargs):
#   h = T/N
#   Y = torch.zeros(
#     size=(N+1, *y.shape), dtype=torch.float32, device=y.device,
#   )

#   Y[0] = y

#   for i in range(N):
#     t = i*h
#     Y[i+1] = Φ(t=t, x=x, y=Y[i], h=h, F=F, **kwargs)

#   return Y

## Debug 2/2 (1/2 in continuous_block_enc_dec):
def solve_sequential(x0, y, N, T, c, Φ, F, **kwargs):
  h = T/N
  x = y

  for i in range(N):
    t = i*h
    x = Φ(t=t, x=x0, y=x, h=h, F=F, **kwargs)

  return x

