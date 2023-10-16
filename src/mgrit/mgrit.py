import torch

def coarse_grid_error_approximation(uΔ, NΔ, Φ, ψ, dt, rΔ=None, **kwargs):
  #if rΔ is not None: uΔ[0] += rΔ[0]  <-- rΔ[0] should always be 0
  for i in range(1, NΔ+1):  # serial for
    uΔ[i] = Φ(F=ψ, i=i-1, x=uΔ[i-1], dt=dt, **kwargs)
    if rΔ is not None: uΔ[i] += rΔ[i]

def compute_r(u, N, Φ, ψ, dt, **kwargs):
  a = torch.empty_like(u)
  a[0] = u[0]
  for i in range(1, N+1):  # parallel for
    a[i] = u[i] - Φ(F=ψ, i=i-1, x=u[i-1], dt=dt, **kwargs)
  
  ## r := g - a, with g[0] = u0, g[1:] = 0
  r = -a  
  _ = r[0].zero_()
  return r

# def correct_u(u, eΔ, NΔ, c):
#   for i in range(NΔ):
#     u[c*i] += eΔ[i]

def F_relaxation(u, N, c, Φ, ψ, dt, r=None, **kwargs):
  for i in range(N//c):  # parallel for
    if r is not None: u[c*i] += r[c*i]

    for ii in range(c-1):  # serial for
      idx = c*i + ii + 1
      u[idx] = Φ(F=ψ, i=idx-1, x=u[idx-1], dt=dt, **kwargs)
      if r is not None: u[idx] += r[idx]

def C_relaxation(u, N, c, Φ, ψ, dt, r=None, **kwargs):
  for i in range(1, N//c + 1):  # parallel for
    idx = c*i
    u[idx] = Φ(F=ψ, i=idx-1, x=u[idx-1], dt=dt, **kwargs)
    if r is not None: u[idx] += r[idx]

def interpolate_u(u, vΔ, NΔ, c):
  for i in range(NΔ):
    u[c*i] = vΔ[i]

def MGRIT_fwd(
    u0, Ns, T, c, Φ, Ψ, relaxation, num_levels, num_iterations, **kwargs
  ):
  with torch.no_grad():
    # u0 = torch.randn_like(x)  # initial guess
    # a = torch.empty(size=(Ns[0]+1, *x.shape))
    u = u0.new(size=(Ns[0]+1, *u0.shape)).zero_()  # randomize?
    u[0] = u0.clone()
    U = num_levels * [None]

    for iteration in range(num_iterations):
      r = None

      ## Relax and go down
      for level in range(num_levels-1):
        N, dt, ψ = obtain_N_dt_ψ(level=level, Ns=Ns, T=T, Ψ=Ψ, c=c)

        if level > 0: u, r = uΔ, rΔ

        relax_approximation(
          u=u, N=N, c=c, Φ=Φ, ψ=ψ, dt=dt, relaxation=relaxation, r=r, **kwargs
        )
        r = compute_r(u=u, N=N, Φ=Φ, ψ=ψ, dt=dt, **kwargs)
        uΔ, rΔ = restrict_to_coarser_grid(u=u, r=r, c=c)

        U[level] = u.clone()
      
      ## Coarsest level
      level = num_levels-1
      N, dt, ψ = obtain_N_dt_ψ(level=level, Ns=Ns, T=T, Ψ=Ψ, c=c)

      vΔ = uΔ.clone()
      coarse_grid_error_approximation(vΔ, N, Φ, ψ, dt, r=rΔ, **kwargs)
      # eΔ = vΔ - uΔ

      ## Correct and go up
      for level in range(num_levels-2, -1, -1):
        N, dt, ψ = obtain_N_dt_ψ(level=level, Ns=Ns, T=T, Ψ=Ψ, c=c)
        u = U[level]

        # correct_u(u=u, eΔ=eΔ, NΔ=N//c, c=c)
        interpolate_u(u=u, vΔ=vΔ, NΔ=N//c, c=c)
        relax_approximation(
          u=u, N=N, c=c, Φ=Φ, ψ=ψ, dt=dt, relaxation='F', **kwargs
        )

        vΔ = u

      # r = compute_r()
      # if r.norm() > tol: break

  return u

def obtain_N_dt_ψ(level, Ns, Ψ, T, c):
  N = Ns[level]
  dt = T/N
  ψ = Ψ[::c**level]#[Ψ[i] for i in range(len(Ψ)) if i % c**level == 0]
  return N, dt, ψ

def relax_approximation(u, N, c, Φ, ψ, dt, relaxation, r=None, **kwargs):
  ## Relax
  if relaxation == 'F':
    F_relaxation(u, N, c, Φ, ψ, dt, r, **kwargs)

  elif relaxation == 'FCF':
    F_relaxation(u, N, c, Φ, ψ, dt, r, **kwargs)
    C_relaxation(u, N, c, Φ, ψ, dt, **kwargs)
    F_relaxation(u, N, c, Φ, ψ, dt, **kwargs)

  else: raise Exception()

def restrict_to_coarser_grid(u, r, c):
  ## Restrict approximation and residual to the next coarser grid
  u = u[::c].clone()
  r = r[::c].clone()
  return u, r



















