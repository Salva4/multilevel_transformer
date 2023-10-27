


def Φ_ForwardEuler(i, F, x, dt, **other_F_inputs):
  F_i_x = F(i, x, **other_F_inputs)['x']

  return x + dt * F_i_x

def Φ_Heun(i, F, x, dt, **other_F_inputs):
  k1 = F(i  , x          , **other_F_inputs)['x']
  k2 = F(i+1, x + dt * k1, **other_F_inputs)['x']

  return x + dt * (k1 + k2)/2

def Φ_RK4(i, F, x, dt, **other_F_inputs):
  k1 = F(i,   x            , **other_F_inputs)['x']
  k2 = F(i+1, x + dt/2 * k1, **other_F_inputs)['x']
  k3 = F(i+1, x + dt/2 * k2, **other_F_inputs)['x']
  k4 = F(i+2, x + dt   * k3, **other_F_inputs)['x']

  return x + dt * (k1 + 2*k2 + 2*k3 + k4)/6

