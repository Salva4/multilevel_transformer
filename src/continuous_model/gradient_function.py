from itertools import chain
import torch
import sys

from .solve.solve_sequential import solve_sequential
from .solve.solve_MGRIT import solve_MGRIT
from .solve.solve_MGOPT import solve_MGOPT

sys.path.append('../')
from ode_solvers.ode_solvers import Φ_ForwardEuler

LAYERS_IDX_CONSTANT = {'Forward Euler': 1, 'Heun': 1, 'RK4': 2}
NUM_LAYERS_INVOLVED = {'Forward Euler': 1, 'Heun': 2, 'RK4': 3}
# NUM_LAYERS_TOTAL = lambda N: {'Forward Euler': N, 'Heun': N+1, 'RK4': 2*N + 1}

class GradientFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, use_mgrit, use_mgopt, fwd_pass_details):
    def F(t, x, **other_F_inputs):
      return ψ[round(t/h*LAYERS_IDX_CONSTANT[solver])](x, **other_F_inputs)

    N = fwd_pass_details['N']
    T = fwd_pass_details['T']
    ψ = fwd_pass_details['ψ']
    solver = fwd_pass_details['solver']

    fwd_pass_details['F'] = F
    h = T/N

    if use_mgrit and use_mgopt: raise Exception()

    if not use_mgrit and \
       not use_mgopt: solve = solve_sequential
    elif   use_mgrit: solve = solve_MGRIT
    elif   use_mgopt: solve = solve_MGOPT

    y = solve(x, **fwd_pass_details)

    ctx.fwd_pass_details = fwd_pass_details
    ctx.h = h
    ctx.solve = solve
    ctx.y = y

    return y[-1]

  @staticmethod
  def backward(ctx, *g):
    def dF(t_inv, g, **other_F_inputs):
      i = round(N - 1 - t_inv/h)
      t = T - h - t_inv

      with torch.enable_grad():
        inputs = y[i]
        inputs.requires_grad_()
        outputs = Φ(t, inputs, h, F) - inputs
        grads = torch.autograd.grad(
          outputs, (inputs,), g, allow_unused=True,
        )

        ## d(Φ^T g)/du = (du/du)^T g + h(dF/du)^T g = g + h(dF/du)^T g
        ## --> (dF/du)^T g = (d(Φ^T g)/du - g)/h
        g = grads[0]/h

      return {'x': g}

    bwd_pass_details = ctx.fwd_pass_details
    h = ctx.h
    solve = ctx.solve
    y = ctx.y

    N = bwd_pass_details['N']
    T = bwd_pass_details['T']
    solver = bwd_pass_details.pop('solver')
    Φ = bwd_pass_details.pop('Φ')
    F = bwd_pass_details.pop('F')
    ψ = bwd_pass_details['ψ']

    # Np = NUM_LAYERS_TOTAL(N)[solver]
    dparameters = [
      [torch.zeros_like(p) for p in ψ[i].parameters()] for i in range(len(ψ))
    ]

    bwd_pass_details['solver'] = 'Forward Euler'
    bwd_pass_details['Φ'] = Φ_ForwardEuler
    bwd_pass_details['F'] = dF

    g = g[0]
    dy = solve(g, **bwd_pass_details)

    ctx.dy = dy

    for i in range(N):
      t = i*h

      with torch.enable_grad():
        inputs = y[i]
        inputs.requires_grad_()
        parameters = []
        for ii in range(i*LAYERS_IDX_CONSTANT[solver], i*LAYERS_IDX_CONSTANT[solver] + NUM_LAYERS_INVOLVED[solver]):
          parameters.extend(list(ψ[ii].parameters()))

        g = dy[N - 1 - i]
        outputs = Φ(t, inputs, h, F) - inputs
        grads = torch.autograd.grad(
          outputs, (*parameters,), g, allow_unused=True,
        )

      ## Assign gradients to parameters
      for j, (p, dp) in enumerate(zip(parameters, grads)):
        if p.grad is not None: p.grad += dp
        else: p.grad = dp

    return dy[-1], None, None, None













