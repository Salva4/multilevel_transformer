import torch

def coarse_grid_error_approximation(uΔ, NΔ, Φ, F, hΔ, rΔ, **kwargs):
  vΔ = uΔ.clone()
  #if rΔ is not None: vΔ[0] += rΔ[0]  <-- rΔ[0] should always be 0
  for i in range(NΔ):  # serial for
    t = i*hΔ
    vΔ[i+1] = Φ(t=t, x=vΔ[i], h=hΔ, F=F, **kwargs) \
              + uΔ[i+1] - Φ(t=t, x=uΔ[i], h=hΔ, F=F, **kwargs) \
              + rΔ[i+1]
  return vΔ

def compute_r(u, N, Φ, F, h, **kwargs):
  a = torch.empty_like(u)
  a[0] = u[0].clone()
  for i in range(N):  # parallel for
    t = i*h
    a[i+1] = u[i+1].clone() - Φ(t=t, x=u[i], h=h, F=F, **kwargs)

  ## r := g - a, with g[0] = u0, g[1:] = 0
  r = -a.clone()
  _ = r[0].zero_()
  return r

def F_relaxation(u, N, c, Φ, F, h, **kwargs):
  for i in range(N//c):  # parallel for
    for ii in range(c-1):  # serial for
      idx = c*i + ii
      t = idx*h
      u[idx+1] = Φ(t=t, x=u[idx], h=h, F=F, **kwargs)

def C_relaxation(u, N, c, Φ, F, h, **kwargs):
  for i in range(1, N//c + 1):  # parallel for
    idx = c*i - 1
    t = idx*h
    u[idx+1] = Φ(t=t, x=u[idx], h=h, F=F, **kwargs)

def interpolate_u(u, eΔ, NΔ, c):
  for i in range(NΔ+1):  # parallel for
    u[c*i] += eΔ[i]

@torch.no_grad()
def MGRIT(u0, N, T, c, Φ, F, relaxation, num_iterations, **kwargs):
  ''' MGRIT implementation with only 2 levels '''

  NΔ = N//c
  h, hΔ = T/N, T/NΔ
  # FΔ = lambda t, x, **kwargs: F(c*t, x, **kwargs)

  u = u0.new(size=(N+1, *u0.shape)).zero_()  # randomize?
  u[0] = u0.clone()

  for iteration in range(num_iterations):

    ## Fine level: relax and go down
    relax_approximation(
      u=u, N=N, c=c, Φ=Φ, F=F, h=h, relaxation=relaxation, **kwargs,
    )
    r = compute_r(u=u, N=N, Φ=Φ, F=F, h=h, **kwargs)
    uΔ, rΔ = restrict_to_coarser_grid(u=u, r=r, c=c)

    ## Coarse level: compute coarse grid approximation
    vΔ = coarse_grid_error_approximation(
      uΔ=uΔ, NΔ=NΔ, Φ=Φ, F=F, hΔ=hΔ, rΔ=rΔ, **kwargs,
    )
    eΔ = vΔ.clone() - uΔ.clone()

    ## Fine level: correct and go up
    interpolate_u(u=u, eΔ=eΔ, NΔ=NΔ, c=c)
    relax_approximation(
      u=u, N=N, c=c, Φ=Φ, F=F, h=h, relaxation='F', **kwargs,
    )

  return u

def relax_approximation(u, N, c, Φ, F, h, relaxation, **kwargs):
  if relaxation == 'F':
    F_relaxation(u, N, c, Φ, F, h, **kwargs)

  elif relaxation == 'FCF':
    F_relaxation(u, N, c, Φ, F, h, **kwargs)
    C_relaxation(u, N, c, Φ, F, h, **kwargs)
    F_relaxation(u, N, c, Φ, F, h, **kwargs)

  else: raise Exception()

def restrict_to_coarser_grid(u, r, c):
  ## Restrict approximation and residual to the next coarser grid
  uΔ = u[::c].clone()
  rΔ = r[::c].clone()
  return uΔ, rΔ


