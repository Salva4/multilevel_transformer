import copy
import torch
import torch.nn as nn
import sys

sys.path.append('../../../src/')

from ode_solvers.ode_solvers import obtain_Φ
from mgrit.mgrit import MGRIT
from continuous_model.gradient_function import GradientFunction

## Debug
from continuous_model.fwd_bwd_pass.solve_sequential import solve_sequential

class ContinuousBlock(nn.Module):
  def __init__(
  self, state_symbol, ψ, N, T, ode_solver, coarsening_factor=2, **kwargs
):
    super().__init__()
    self.state_symbol = state_symbol
    self.N = N
    self.T = T
    self.c = c = coarsening_factor
    self.ode_solver = ode_solver
    self.Φ = obtain_Φ(ode_solver)

    self.ψ = nn.ModuleList([])

    if self.ode_solver == 'Forward Euler':
      for i in range(self.N): self.ψ.append(ψ[i])  # basis functions

    elif self.ode_solver == 'Heun':
      for i in range(self.N): self.ψ.append(ψ[i])
      self.ψ.append(copy.deepcopy(ψ[-1]))

    elif self.ode_solver == 'RK4':
      for i in range(self.N):
        self.ψ.append(ψ[i])
        self.ψ.append(copy.deepcopy(ψ[i]))
      self.ψ.append(copy.deepcopy(ψ[-1]))

  def forward(self, x, level=0, use_mgrit=False, **fwd_pass_details):
    output = {}

    N = self.N // self.c**level
    T = self.T
    c = self.c
    Φ = self.Φ
    ψ = self.ψ[::c**level]

    if fwd_pass_details.get('ode_solver', self.ode_solver) != self.ode_solver:
      ode_solver = fwd_pass_details['ode_solver']
      Φ = obtain_Φ(ode_solver)

      if self.ode_solver == 'Forward Euler' or \
        (self.ode_solver == 'Heun' and ode_solver == 'RK4'):
        raise Exception('New ODE solver cannot be more complex than the default one.')

      if self.ode_solver == 'Heun':  #(ode_solver == 'Forward Euler')
        ψ = ψ[:-1]
      elif ode_solver == 'Heun':  # (self.ode_solver = 'RK4')
        ψ = ψ[::2]
      elif ode_solver == 'Forward Euler':  # (self.ode_solver == 'RK4')
        ψ = ψ[:-1:2]

    else: ode_solver = self.ode_solver

    assert N > 0, f'Level {level} incompatible with {self.N} layers at the finest level and a coarsening factor of {self.c}'

    ode_fwd_details = {
      'N': N, 'T': T, 'c': c, 'solver': ode_solver, 'Φ': Φ, 'ψ': ψ,
    }
    fwd_pass_details.update(ode_fwd_details)

    ## OPTION (I): my implementation of the backward pass via 
    ##...torch.autograd.function. SLOW but works. Only used when comparing the
    ##...MGRIT method, as usual PyTorch implementation is incompatible with
    ##...MGRIT.
    # x = GradientFunction.apply(x, use_mgrit, fwd_pass_details)


    ## OPTION (II): PyTorch implementation. Faster but incompatible with
    ##...MGRIT.
    h = T/N
    LAYERS_IDX_CONSTANT = {'Forward Euler': 1, 'Heun': 1, 'RK4': 2}
    def F(t, x, **other_F_inputs):
      return ψ[round(t/h*LAYERS_IDX_CONSTANT[ode_solver])](x, **other_F_inputs)
    fwd_pass_details['F'] = F
    x = solve_sequential(x, **fwd_pass_details)

    ## Regardless of which is the selected option, the solve_sequential.py
    ##....file must be commented/uncommented correspondingly.

    output['x'] = x

    return output

  def interpolate_weights(self, fine_level, interpolation):
    c = self.c
    ψ_fine = self.ψ[::c**fine_level]
    # ψ_coarse = self.ψ[::c**(fine_level+1)]

    if interpolation == 'constant':
      print(f'Applying constant weights interpolation.')
      
      num_coarse_nodes = (len(ψ_fine) - 1)//c + 1
      for i in range(num_coarse_nodes):
        _ψ_coarse = ψ_fine[c*i]

        for ii in range(1, c):
          if c*i + ii == len(ψ_fine): break

          _ψ_to_interpolate = ψ_fine[c*i + ii]

          for p_c, p_to_interpolate in zip(
            _ψ_coarse        .parameters(),
            _ψ_to_interpolate.parameters(),
          ):
            p_to_interpolate.data = p_c.data.clone()

    elif interpolation == 'linear':
      print(f'Applying linear weights interpolation.')

      num_coarse_nodes = (len(ψ_fine) - 1)//c + 1
      for i in range(num_coarse_nodes - 1):
        _ψ_coarse1 = ψ_fine[c*i]
        _ψ_coarse2 = ψ_fine[c*(i+1)]

        for ii in range(1, c):
          if c*i + ii == len(ψ_fine): break

          _ψ_to_interpolate = ψ_fine[c*i + ii]

          for p_c1, p_c2, p_to_interpolate in zip(
            _ψ_coarse1       .parameters(),
            _ψ_coarse2       .parameters(),
            _ψ_to_interpolate.parameters(),
          ):
            p_to_interpolate.data = \
              (1 - ii/c)*p_c1.data.clone() + (ii/c)*p_c2.data.clone()

      ## If there aren't any more coarse nodes, don't extrapolate but copy.
      i = num_coarse_nodes - 1
      _ψ_last_coarse_node = ψ_fine[c*i]

      for ii in range(1, c):
        if c*i + ii == len(ψ_fine): break

        _ψ_to_interpolate = ψ_fine[c*i + ii]

        for p_c_last, p_to_interpolate in zip(
          _ψ_last_coarse_node.parameters(),
          _ψ_to_interpolate  .parameters(),
        ):
          p_to_interpolate.data = p_c_last.data.clone()

    else: raise Exception()  # future work: quadratic splines & cubic splines




