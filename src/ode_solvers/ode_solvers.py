

## F's
def F_ForwardEuler(t, x, h, F, **other_F_inputs):
  k1 = F(t, x, **other_F_inputs)['x']
  return k1

def F_Heun(t, x, h, F, **other_F_inputs):
  k1 = F(t    , x       , **other_F_inputs)['x']
  k2 = F(t + h, x + h*k1, **other_F_inputs)['x']
  return (k1 + k2)/2

def F_RK4(t, x, h, F, **other_F_inputs):
  k1 = F(t      , x         , **other_F_inputs)['x']
  k2 = F(t + h/2, x + h/2*k1, **other_F_inputs)['x']
  k3 = F(t + h/2, x + h/2*k2, **other_F_inputs)['x']
  k4 = F(t + h  , x + h  *k3, **other_F_inputs)['x']
  return (k1 + 2*k2 + 2*k3 + k4)/6

## Φ's
def Φ_ForwardEuler(t, x, h, F, **other_F_inputs):
  F = F_ForwardEuler(t, x, h, F, **other_F_inputs)
  return x + h*F

def Φ_Heun(t, x, h, F, **other_F_inputs):
  F = F_Heun(t, x, h, F, **other_F_inputs)
  return x + h*F

def Φ_RK4(t, x, h, F, **other_F_inputs):
  F = F_RK4(t, x, h, F, **other_F_inputs)
  return x + h*F

## Handler
def obtain_Φ(solver):
  if   solver == 'Forward Euler': return Φ_ForwardEuler
  elif solver == 'Heun'         : return Φ_Heun
  elif solver == 'RK4'          : return Φ_RK4
  else: raise Exception('Unknown solver.')




