import copy
import torch
import torch.nn as nn

class ContinuousUnit(nn.Module):
  def __init__(self, module, N, T, scheme='Euler'):
    super().__init__()
    self.N, self.T = N, T
    self.phi = nn.ModuleList([copy.deepcopy(module) \
                              for _ in range(N+1)])
    self.dt = T/N
    self.scheme = scheme
    assert scheme in 'Euler Heun RK4'.split()

  def forward(self, x, **kwargs):
    N, dt, scheme, phi = self.N, self.dt, self.scheme, self.phi

    if scheme == 'Euler':  # x_{t+1} = x_{t} + dt·F(x_{t}, theta_{t})
      h = dt
      for n in range(N):
        x = x + h*phi[n](x, **kwargs)

    elif scheme == 'Heun':  # x_{t+1}' = x_{t} + dt·F(x_{t}, theta_{t})
                            # x_{t+1} = x_{t} + dt/2·(F(x_{t}, theta_{t}) + F(x_{t+1}', theta_{t+1}))
      h = dt
      for n in range(N):
        phi_t_x = phi[n](x, **kwargs)
        x_tilde = x + h*phi_t_x    # with grad | no_grad?
        x = x + h/2*(phi_t_x + phi[n+1](x_tilde, **kwargs))

    elif scheme == 'RK4':
      h = 2*dt
      for n in range(0, N, 2):
        phi_t, phi_tp05, phi_tp1 = phi[n], phi[n+1], phi[n+2]
        k1 = phi[n]  (x           , **kwargs)
        k2 = phi[n+1](x + 1/2*h*k1, **kwargs)
        k3 = phi[n+1](x + 1/2*h*k2, **kwargs)
        k4 = phi[n+2](x +     h*k3, **kwargs)
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4)

    # elif scheme == 'debug':
    #   h = dt
    #   for n in range(N):
    #     x = x + h*phi[n](x, **kwargs)    

    return x































