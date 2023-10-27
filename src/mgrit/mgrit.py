import torch

def coarse_grid_error_approximation(uΔ, NΔ, Φ, ψ, hΔ, rΔ, **kwargs):
  vΔ = uΔ.clone()
  #if rΔ is not None: vΔ[0] += rΔ[0]  <-- rΔ[0] should always be 0
  for i in range(NΔ):  # serial for
    vΔ[i+1] = Φ(F=ψ, i=i, x=vΔ[i], dt=hΔ, **kwargs) \
              + uΔ[i+1] - Φ(F=ψ, i=i, x=uΔ[i], dt=hΔ, **kwargs) \
              + rΔ[i+1]
  return vΔ

def compute_r(u, N, Φ, ψ, h, **kwargs):
  a = torch.empty_like(u)
  a[0] = u[0].clone()
  for i in range(N):  # parallel for
    a[i+1] = u[i+1].clone() - Φ(F=ψ, i=i, x=u[i], dt=h, **kwargs)
  
  ## r := g - a, with g[0] = u0, g[1:] = 0
  r = -a.clone()
  _ = r[0].zero_()
  return r

def F_relaxation(u, N, c, Φ, ψ, h, **kwargs):
  for i in range(N//c):  # parallel for
    for ii in range(c-1):  # serial for
      idx = c*i + ii
      u[idx+1] = Φ(F=ψ, i=idx, x=u[idx], dt=h, **kwargs)

def C_relaxation(u, N, c, Φ, ψ, h, **kwargs):
  for i in range(1, N//c + 1):  # parallel for
    idx = c*i - 1
    u[idx+1] = Φ(F=ψ, i=idx, x=u[idx], dt=h, **kwargs)

def interpolate_u(u, eΔ, NΔ, c):
  for i in range(NΔ+1):  # parallel for
    u[c*i] += eΔ[i]

@torch.no_grad()
def MGRIT(u0, N, T, c, Φ, F, relaxation, num_iterations, **kwargs):
  ''' MGRIT implementation with only 2 levels '''

  NΔ = N//c
  h, hΔ = T/N, T/NΔ
  # ψ, ψΔ = F, F[::c]
  ψ, ψΔ = F, lambda i, x: F(c*i, x)

  u = u0.new(size=(N+1, *u0.shape)).zero_()  # randomize?
  u[0] = u0.clone()

  for iteration in range(num_iterations):

    ## Fine level: relax and go down
    relax_approximation(
      u=u, N=N, c=c, Φ=Φ, ψ=ψ, h=h, relaxation=relaxation, **kwargs,
    )
    r = compute_r(u=u, N=N, Φ=Φ, ψ=ψ, h=h, **kwargs)
    uΔ, rΔ = restrict_to_coarser_grid(u=u, r=r, c=c)
    
    ## Coarse level: compute coarse grid approximation
    vΔ = coarse_grid_error_approximation(
      uΔ=uΔ, NΔ=NΔ, Φ=Φ, ψ=ψΔ, hΔ=hΔ, rΔ=rΔ, **kwargs,
    )
    eΔ = vΔ.clone() - uΔ.clone()

    ## Fine level: correct and go up
    interpolate_u(u=u, eΔ=eΔ, NΔ=NΔ, c=c)
    relax_approximation(
      u=u, N=N, c=c, Φ=Φ, ψ=ψ, h=h, relaxation='F', **kwargs,
    )

  return u

def relax_approximation(u, N, c, Φ, ψ, h, relaxation, **kwargs):
  if relaxation == 'F':
    F_relaxation(u, N, c, Φ, ψ, h, **kwargs)

  elif relaxation == 'FCF':
    F_relaxation(u, N, c, Φ, ψ, h, **kwargs)
    C_relaxation(u, N, c, Φ, ψ, h, **kwargs)
    F_relaxation(u, N, c, Φ, ψ, h, **kwargs)

  else: raise Exception()

def restrict_to_coarser_grid(u, r, c):
  ## Restrict approximation and residual to the next coarser grid
  uΔ = u[::c].clone()
  rΔ = r[::c].clone()
  return uΔ, rΔ



















