import torch

from .solve.solve_sequential import solve_sequential
from .solve.solve_MGRIT import solve_MGRIT
from .solve.solve_MGOPT import solve_MGOPT

class GradientFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, extended_kwargs, use_MGRIT, use_MGOPT):
    def F(i, x, **other_F_inputs): return ψ[i](x, **other_F_inputs)

    (N, T, c, Φ, ψ) = [extended_kwargs.pop(s) for s in 'N T c Φ ψ'.split()]
    kwargs = extended_kwargs
    (ctx.x, ctx.N, ctx.T, ctx.c, ctx.Φ, ctx.ψ, ctx.kwargs, ctx.F) = (
      x, N, T, c, Φ, ψ, kwargs, F,
    )

    if not use_MGRIT and not use_MGOPT: solve = solve_sequential
    elif use_MGRIT: solve = solve_MGRIT
    elif use_MGOPT: solve = solve_MGOPT
    if use_MGRIT and use_MGOPT: raise Exception()

    y = solve(x, N, T, c, Φ, F=F, **kwargs)

    (ctx.y, ctx.solve) = (y, solve)

    return y[-1]

  @staticmethod
  def backward(ctx, *g):
    def dF(ip, g, **other_F_inputs):
      i = N - 1 - ip
      with torch.enable_grad():
        inputs = y[i]
        inputs.requires_grad_()
        parameters = ψ[i].parameters()
        outputs = F(i, inputs, **other_F_inputs)['x']#ψ[i](inputs)
        g, *dparameters = torch.autograd.grad(
          outputs, (inputs, *parameters), g, allow_unused=True,
        )
      return {'x': g}

    (y, N, T, c, Φ, ψ, kwargs, F, solve) = (
      ctx.y, ctx.N, ctx.T, ctx.c, ctx.Φ, ctx.ψ, ctx.kwargs, ctx.F, ctx.solve,
    )

    dparameters = [[None for p in ψ[i].parameters()] for i in range(N)]

    g = g[0]
    dy = solve(g, N, T, c, Φ, F=dF, **kwargs)

    ctx.dy = dy

    ## Assign gradients
    for i in range(N):
      for j, p in enumerate(ψ[i].parameters()):
        p.grad = dparameters[i][j]

    return dy[-1], None, None, None, None, None, None, None, None





























