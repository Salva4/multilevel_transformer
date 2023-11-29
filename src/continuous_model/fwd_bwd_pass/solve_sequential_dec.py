import torch

## No-debug:
@torch.no_grad()
def solve_sequential(x, y, N, T, c, Φ, F, **kwargs):#state_symbol, **kwargs):
  # print('solve_sequential')
  h = T/N
  Y = torch.zeros(
    size=(N+1, *y.shape), dtype=torch.float32, device=y.device,
  )

  Y[0] = y

  for i in range(N):
    t = i*h
    Y[i+1] = Φ(t=t, x=x, y=Y[i], h=h, F=F, **kwargs)

  return Y

# ## Debug:
# def solve_sequential(x0, N, T, c, Φ, F, **kwargs):
#   h = T/N
#   x = torch.zeros(
#     size=(N+1, *x0.shape), dtype=torch.float32, device=x0.device,
#   )

#   x = x0

#   for i in range(N):
#     t = i*h
#     x = Φ(t=t, x=x, h=h, F=F, **kwargs)

#   return x

