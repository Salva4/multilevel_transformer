import copy
import torch
import torch.nn as nn
import sys

sys.path.append('../../../src/')

from ode_solvers.ode_solvers import obtain_Φ#Φ_ForwardEuler, Φ_Heun, Φ_RK4
from mgrit.mgrit import MGRIT
from continuous_model.gradient_function import GradientFunction

## Debug
from continuous_model.fwd_bwd_pass.solve_sequential import solve_sequential

class ContinuousBlock(nn.Module):
  def __init__(self, ψ, N, T, ode_solver, coarsening_factor=2, **kwargs):#, num_levels):#, interpol):
    super().__init__()
    self.N = N
    self.T = T
    self.c = c = coarsening_factor
    self.ode_solver = ode_solver
    self.Φ = obtain_Φ(ode_solver)
    # self.num_levels = num_levels
    # self.interpol = interpol

    # self.Ns = []
    # level = 0
    # while Nf % c**level == 0:
    #   self.Ns.append(Nf // c**level)
    #   level += 1

    self.ψ = nn.ModuleList([])

    if self.ode_solver == 'Forward Euler':
      # self.Φ = Φ_ForwardEuler
      for i in range(self.N): self.ψ.append(ψ[i])  # basis functions

    elif self.ode_solver == 'Heun':
      # self.Φ = Φ_Heun
      for i in range(self.N): self.ψ.append(ψ[i])
      self.ψ.append(copy.deepcopy(ψ[-1]))

    elif self.ode_solver == 'RK4':
      # self.Φ = Φ_RK4
      for i in range(self.N):
        self.ψ.append(ψ[i])
        self.ψ.append(copy.deepcopy(ψ[i]))
      self.ψ.append(copy.deepcopy(ψ[-1]))

    # self.weights = [
    #   'fc1.weight', 'fc1.bias',
    #   'fc2.weight', 'fc2.bias',
    #   'att.in_proj_weight', 'att.in_proj_bias', 'att.out_proj.weight', 'att.out_proj.bias',
    #   'ln1.weight', 'ln1.bias',
    #   'ln2.weight', 'ln2.bias'
    # ]

  def forward(self, x, level=0, use_mgrit=False, **fwd_pass_details):
    output = {}

    N = self.N // self.c**level  #self.Ns[level]
    T = self.T
    c = self.c
    Φ = self.Φ
    ode_solver = self.ode_solver
    ψ = self.ψ[::c**level]

    ode_fwd_details = {
      'N': N, 'T': T, 'c': c, 'solver': ode_solver, 'Φ': Φ, 'ψ': ψ,
    }
    fwd_pass_details.update(ode_fwd_details)

    ## No-debug:
    x = GradientFunction.apply(x, use_mgrit, fwd_pass_details)

    ## Debug:
    # h = T/N
    # LAYERS_IDX_CONSTANT = {'Forward Euler': 1, 'Heun': 1, 'RK4': 2}
    # def F(t, x, **other_F_inputs):
    #   return ψ[round(t/h*LAYERS_IDX_CONSTANT[ode_solver])](x, **other_F_inputs)
    # fwd_pass_details['F'] = F
    # x = solve_sequential(x, **fwd_pass_details)

    output['x'] = x

    return output

  def interpolate_weights(self, level, interpolation):
    c = self.c
    ψ_fine = self.ψ[::c**level]
    # ψ_coarse = self.ψ[::c**(level+1)]

    if interpolation == 'constant':
      for i in range(c, len(ψ_fine), c):
        _ψ_coarse1 = ψ_fine[i-c]
        # _ψ_coarse2 = ψ_fine[i]

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



