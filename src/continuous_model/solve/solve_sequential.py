import torch

@torch.no_grad()
def solve_sequential(x0, N, T, c, Φ, F, **kwargs):
  dt = T/N
  x = torch.zeros(
    size=(N+1, *x0.shape), dtype=torch.float32, device=x0.device,
  )
  x[0] = x0

  for i in range(N):
    x[i+1] = Φ(F=F, i=i, x=x[i], dt=dt, **kwargs)

  return x


