import copy
import torch
import torch.nn as nn
import sys

sys.path.append('../../../src/')

from ode_solvers.ode_solvers import obtain_Φ#Φ_ForwardEuler, Φ_Heun, Φ_RK4
from ode_solvers.ode_solvers_dec import obtain_Φ as obtain_Φ_dec
from mgrit.mgrit import MGRIT
from continuous_model.gradient_function_enc import GradientFunction \
                                                as GradientFunction_enc
from continuous_model.gradient_function_dec import GradientFunction \
                                                as GradientFunction_dec

## Debug
from continuous_model.fwd_bwd_pass.solve_sequential import solve_sequential
from continuous_model.fwd_bwd_pass.solve_sequential_dec \
                              import solve_sequential as solve_sequential_dec

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
    self.Φ = obtain_Φ(ode_solver) if state_symbol=='x' else \
             obtain_Φ_dec(ode_solver)

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

  def forward(self, x, y, level=0, use_mgrit=False, **fwd_pass_details):
    output = {}

    N = self.N // self.c**level
    T = self.T
    c = self.c
    Φ = self.Φ
    ode_solver = self.ode_solver
    ψ = self.ψ[::c**level]

    ode_fwd_details = {
      'N': N, 'T': T, 'c': c, 'solver': ode_solver, 'Φ': Φ, 'ψ': ψ,
    }
    fwd_pass_details.update(ode_fwd_details)

    ## OPTION (I), corresponding to OPTION (I) in continuous_block.py. 
    # gradient_fn = GradientFunction_enc if self.state_symbol == 'x' else \
    #               GradientFunction_dec
    # x = gradient_fn.apply(x, y, use_mgrit, fwd_pass_details)

    ## OPTION (II), corresponding to OPTION (II) in continuous_block.py.
    _solve_sequential = solve_sequential if self.state_symbol == 'x' else \
                        solve_sequential_dec
    h = T/N
    LAYERS_IDX_CONSTANT = {'Forward Euler': 1, 'Heun': 1, 'RK4': 2}
    def F(t, x, y, **other_F_inputs):
      return ψ[round(t/h*LAYERS_IDX_CONSTANT[ode_solver])](
        x=x, y=y, **other_F_inputs,
      )
    fwd_pass_details['F'] = F
    x = _solve_sequential(x0=x, y=y, **fwd_pass_details)

    ## Regardless of which is the selected option, the solve_sequential.py
    ##....file must be commented/uncommented correspondingly.

    output[self.state_symbol] = x

    return output

  def interpolate_weights(self, level, interpolation):
    c = self.c
    ψ_fine = self.ψ[::c**level]

    if interpolation == 'constant':
      for i in range(c, len(ψ_fine), c):
        _ψ_coarse1 = ψ_fine[i-c]

        for ii in range(1, c):
          _ψ_to_interpolate = ψ_fine[i - c + ii]
          for p_c1, p_to_interpolate in zip(
            _ψ_coarse1.parameters(), _ψ_to_interpolate.parameters(),
          ):
            p_to_interpolate.data = p_c1.data

      while i < len(ψ_fine) - 1:
        for p_last, p_to_interpolate in zip(
          ψ_fine[i].parameters(), ψ_fine[i+1].parameters(),
        ):
          p_to_interpolate.data = p_last.data

        i += 1

    elif interpolation == 'linear':
      for i in range(c, len(ψ_fine), c):
        _ψ_coarse1 = ψ_fine[i-c]
        _ψ_coarse2 = ψ_fine[i]

        for ii in range(1, c):
          _ψ_to_interpolate = ψ_fine[i - c + ii]
          for p_c1, p_c2, p_to_interpolate in zip(
            _ψ_coarse1.parameters(),
            _ψ_coarse2.parameters(),
            _ψ_to_interpolate.parameters(),
          ):
            p_to_interpolate.data = (1 - ii/c)*p_c1.data + (ii/c)*p_c2.data

      while i < len(ψ_fine) - 1:
        for p_lastbutone, p_last, p_to_interpolate in zip(
          ψ_fine[i-1].parameters(),
          ψ_fine[i  ].parameters(),
          ψ_fine[i+1].parameters(),
        ):
          p_to_interpolate.data = \
            p_last.data + (p_last.data - p_lastbutone.data)

        i += 1

    else: raise Exception()  # add: quadratic splines & cubic splines




