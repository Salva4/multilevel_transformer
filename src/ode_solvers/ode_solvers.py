

def Φ_ForwardEuler(i, F, x, dt, **other_F_inputs):
  F_i = F[i]
  return x + dt * F_i(x, **other_F_inputs)['x']

def Φ_Heun(i, F, x, dt, **other_F_inputs):
  F_i, F_ip1 = F[i], F[i+1]
  k1 = F_i  (x          , **other_F_inputs)['x']
  k2 = F_ip1(x + dt * k1, **other_F_inputs)['x']

  return x + dt * (k1 + k2)/2

def Φ_RK4(i, F, x, dt, **other_F_inputs):
  F_i, F_ip1, F_ip2 = F[i], F[i+1], F[i+2]
  k1 = F_i   (x            , **other_F_inputs)['x']
  k2 = F_ip1 (x + dt/2 * k1, **other_F_inputs)['x']
  k3 = F_ip1 (x + dt/2 * k2, **other_F_inputs)['x']
  k4 = F_ip2 (x + dt   * k3, **other_F_inputs)['x']

  return x + dt * (k1 + 2*k2 + 2*k3 + k4)/6

