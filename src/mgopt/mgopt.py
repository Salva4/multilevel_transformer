import numpy as np
import sys

sys.path.append('..')
from src_utils.filter_dict import filter_keys

def fwd_pass(model, model_inputs, compute_accuracy_if_pertinent=False):
  compute_accuracy = compute_accuracy_if_pertinent \
                 and model_inputs.get('compute_accuracy', False)
  model_outputs = model(
    compute_accuracy=compute_accuracy, 
    **filter_keys(model_inputs, ('compute_accuracy',))
  )
  loss    = model_outputs['loss'   ]
  correct = model_outputs['correct'] if compute_accuracy else None
  total   = model_outputs['total'  ] if compute_accuracy else None

  return loss, (correct, total)

def bwd_pass(model, optimizer, loss, dldΘ, level):
  optimizer.zero_grad()
  loss.backward()
  apply_first_order_correction(model, dldΘ, level)
  optimizer.step()

def train_miniepoch(
  model, optimizer, prepare_inputs, num_batches, level, dldΘ,
  compute_accuracy_if_pertinent=False,
):
  losses = []
  correct, total = 0, 0
  for batch_idx in range(num_batches):
    model_inputs, get_batch_time = prepare_inputs()
    model_inputs['level'] = level
    loss, (_correct, _total) = fwd_pass(
      model, model_inputs, compute_accuracy_if_pertinent
    )
    losses.append(loss.item())

    if _correct is not None:
      correct += _correct
      total   += _total

    bwd_pass(model, optimizer, loss, dldΘ, level)

  if total == 0: correct, total = None, None

  return np.mean(losses), (correct, total)

def obtain_gradient_wrt_parameters(
  model, optimizer, prepare_inputs, num_batches, level,
):
  optimizer.zero_grad()
  for batch_idx in range(num_batches):
    model_inputs, get_batch_time = prepare_inputs()
    model_inputs['level'] = level
    loss, _ = fwd_pass(model, model_inputs)
    loss /= num_batches
    loss.backward()

  for continuous_block in model.continuous_blocks:
    c = continuous_block.c
    for _ψ in continuous_block.ψ[::c**level]:
      for p in _ψ.parameters():
        yield p.grad

def apply_first_order_correction(model, dldΘ, level):
  if dldΘ is None: return
  for continuous_block in model.continuous_blocks:
    for _ψ in continuous_block.ψ[::c**level]:
      for p, dldθ in zip(_ψ.parameters(), dldΘ):
        p.grad += dldθ

def run_cycle(
  model, optimizer, prepare_inputs, num_batches, mu, nu, num_levels,
  multilevel_interpolation,
):
  dldΘ_register = [None]*num_levels
  for level in range(num_levels - 1):
    for presmoothing_iteration in range(mu):
      _ = train_miniepoch(
        model, optimizer, prepare_inputs, num_batches, level,
        dldΘ_register[level], compute_accuracy_if_pertinent=False,
      )
    dldΘ_register[level + 1] = obtain_gradient_wrt_parameters(
      model, optimizer, prepare_inputs, num_batches, level,
    )

  ## Coarsest level
  for coarse_iteration in range(mu):
    _ = train_miniepoch(
      model, optimizer, prepare_inputs, num_batches, level,
      dldΘ_register[level], compute_accuracy_if_pertinent=False,
    )

  for level in range(num_levels - 2, -1, -1):
    ## Correction implicitly done: in our case, interpolation matrix is 
    ##...the identity for even nodes and 0 for the odd nodes. However, the
    ##...odd nodes are not modified at the coarse level.
    model.interpolate_weights(level, multilevel_interpolation)

    for postsmoothing_iteration in range(nu):
      loss, (correct, total) = train_miniepoch(
        model, optimizer, prepare_inputs, num_batches, level,
        dldΘ_register[level], compute_accuracy_if_pertinent=True,
      )

  return loss, (correct, total)

def _MGOPT(
  mgopt_mu, mgopt_nu, mgopt_num_levels, mgopt_cycle, mgopt_num_iterations,
  *args, **kwargs,
):
  return MGOPT(
    mu=mgopt_mu, nu=mgopt_nu, num_levels=mgopt_num_levels, cycle=mgopt_cycle,
    num_iterations=mgopt_num_iterations, *args, **kwargs,
  )

def MGOPT(
  model, optimizer, prepare_inputs, num_batches, mu, nu, num_levels, cycle,
  multilevel_interpolation, num_iterations, losses, accuracy_counter, 
  **kwargs,
):
  model.train()

  if cycle != 'V': raise Exception('F- and W- cycles not implemented yet.')

  for iteration in range(num_iterations):
    loss, (correct, total) = run_cycle(
      model, optimizer, prepare_inputs, num_batches, mu, nu, num_levels,
      multilevel_interpolation,
    )
    losses.append(loss.item())

    if correct is not None:
      accuracy_counter.correct += correct
      accuracy_counter.total   += total




