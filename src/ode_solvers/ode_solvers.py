

def Φ_ForwardEuler(F, x, dt, **other_F_inputs):
  return x + dt * F(x, **other_F_inputs)['x']

def Φ_Heun(F_i, F_ip1, x, dt, **other_F_inputs):
  k1 = F_i  (x          , **other_F_inputs)['x']
  k2 = F_ip1(x + dt * k1, **other_F_inputs)['x']

  return x + dt * (k1 + k2)/2

def Φ_RK4(F_i, F_ipc5, F_ip1, x, dt, **other_F_inputs):
  k1 = F_i   (x            , **other_F_inputs)['x']
  k2 = F_ipc5(x + dt/2 * k1, **other_F_inputs)['x']
  k3 = F_ipc5(x + dt/2 * k2, **other_F_inputs)['x']
  k4 = F_ip1 (x + dt   * k3, **other_F_inputs)['x']

  return x + dt * (k1 + 2*k2 + 2*k3 + k4)/6

