import numpy as np

def fwd_pass(model, model_inputs):
  model_outputs = model(**model_inputs)
  loss = model_outputs['loss']
  return loss

def bwd_pass(model, optimizer, loss, dldΘ, level, c):
  optimizer.zero_grad()
  loss.backward()
  apply_first_order_correction(model, dldΘ, level, c)
  optimizer.step()

def train_miniepoch(
  model, optimizer, prepare_inputs, num_batches, level, c, dldΘ,
):
  losses = []
  for batch_idx in range(num_batches):
    model_inputs, get_batch_time = prepare_inputs()
    model_inputs['level'] = level
    loss = fwd_pass(model, model_inputs)
    losses.append(loss.item())
    bwd_pass(model, optimizer, loss, dldΘ, level, c)

  return np.mean(losses)

def obtain_gradient_wrt_parameters(
  model, optimizer, prepare_inputs, num_batches, level, c,
):
  optimizer.zero_grad()
  for batch_idx in range(num_batches):
    model_inputs, get_batch_time = prepare_inputs()
    model_inputs['level'] = level
    loss = fwd_pass(model, model_inputs)
    loss /= num_batches
    loss.backward()

  for _ψ in model.continuous_block.ψ[::c**level]:
    for p in _ψ.parameters():
      yield p.grad

def apply_first_order_correction(model, dldΘ, level, c):
  if dldΘ is None: return
  for _ψ in model.continuous_block.ψ[::c**level]:
    for p, dldθ in zip(_ψ.parameters(), dldΘ):
      p.grad += dldθ

def run_cycle(
  model, optimizer, prepare_inputs, num_batches, mu, nu, num_levels, c,
):
  dldΘ_register = [None]*num_levels
  for level in range(num_levels - 1):
    for presmoothing_iteration in range(mu):
      _ = train_miniepoch(
        model, optimizer, prepare_inputs, num_batches, level, c,
        dldΘ_register[level],
      )
    dldΘ_register[level + 1] = obtain_gradient_wrt_parameters(
      model, optimizer, prepare_inputs, num_batches, level, c,
    )

  ## Coarsest level
  for coarse_iteration in range(mu):
    _ = train_miniepoch(
      model, optimizer, prepare_inputs, num_batches, level, c,
      dldΘ_register[level],
    )

  for level in range(num_levels - 2, -1, -1):
    for postsmoothing_iteration in range(nu):
      loss = train_miniepoch(
        model, optimizer, prepare_inputs, num_batches, level, c,
        dldΘ_register[level],
      )

  return loss

def _MGOPT(
  mgopt_mu, mgopt_nu, mgopt_num_levels, mgopt_num_iterations, *args, **kwargs,
):
  return MGOPT(
    mu=mgopt_mu, nu=mgopt_nu, num_levels=mgopt_num_levels,
    num_iterations=mgopt_num_iterations, *args, **kwargs,
  )

def MGOPT(
  model, optimizer, prepare_inputs, num_batches, mu, nu, num_levels,
  num_iterations, losses, **kwargs,
):
  c = model.continuous_block.c
  model.train()

  for iteration in range(num_iterations):
    loss = run_cycle(
      model, optimizer, prepare_inputs, num_batches, mu, nu, num_levels, c,
    )
    losses.append(loss.item())


