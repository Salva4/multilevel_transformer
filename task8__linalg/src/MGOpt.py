import copy
import numpy as np
import time
import torch
import torch.nn as nn

from interpolate_MGOpt import interpolate_weights, restrict_weights

PRINT = False

## Auxiliary functions ##########################
def _print(*args):
  if PRINT: print(*args)

def add_τ(_vars, lvl):
  if lvl == _vars.FINE: return
  model, τ = _vars.models[lvl], _vars.τ
  for (p, g) in zip(model.parameters(), τ[lvl]): p.grad.data += g

def init_τ(_vars):
  τ = _vars.τ = {}  
  for lvl in range(_vars.n_lvls):
    τ[lvl] = []
    model = _vars.models[lvl]
    for p_m in model.parameters():
      τ[lvl].append(torch.zeros_like(p_m))

def init_dl_dθ(_vars):
  dl_dθ = _vars.dl_dθ = {}  
  for lvl in range(_vars.n_lvls):
    dl_dθ[lvl] = []
    model = _vars.models[lvl]
    for p_m in model.parameters():
      dl_dθ[lvl].append(torch.zeros_like(p_m))

def fwd(_vars, model, batch, bwd, preds=False):
  ## Forward
  sources, targets = batch
  targets_input, targets_output = targets[:, :-1], targets[:, 1:]
  outputs = model(sources, targets_input)  # (batch, len_max_targets, alphabet)
  outputs_alphabet_dim1 = outputs.transpose(1, 2)  # (batch, alphabet, len_max_targets)
  loss = _vars.loss_function(outputs_alphabet_dim1, targets_output)
  if bwd: loss.backward()  # backward

  ## Monitoring
  if preds:
    _vars.predictions = outputs.argmax(dim=2)
    _vars.correct_training += torch.logical_or(
        _vars.predictions == targets_output, 
        targets_output == _vars.vocabulary_target.pad_id
    ).prod(dim=1).sum(dim=0).item()
    _vars.total_training += targets_output.size()[0]
    
    _vars.sources, _vars.targets_output = sources, targets_output

  return loss

# def get_batches(_vars, n):
#   ## Batch gradient descent
#   batch = next(_vars.iter_dl_tr, None)
#   if batch is None: 
#     _vars.iter_dl_tr = iter(_vars.dl_tr)
#     batch = next(_vars.iter_dl_tr)
#   batches.append(batch)

#   ## SGD
#   # iter_dl = iter(_vars.dl_tr)
#   # batches = [next(iter_dl) for _ in range(n)]

#   ## Batch GD (all dataset)
#   # batches = _vars.dl_tr

#   ## Batch GD (equivalent monitoring to conv)
#   # batches = []
#   # for _ in range(_vars.n_monitoring):
#   #   batch = next(_vars.iter_dl_tr, None)
#   #   if batch is None: 
#   #     _vars.iter_dl_tr = iter(_vars.dl_tr)
#   #     batch = next(_vars.iter_dl_tr)
#   #   batches.append(batch)

#   return batches

def line_search(_vars, model, lvl):
  model_copy = copy.deepcopy(model)
  α = _vars.α = 1
  losses = _vars.losses

  with torch.no_grad():
    while True:
      _print(f'Interpolating weights from level {lvl-1} to level {lvl}' \
          + f' with a factor α of {α}') ## maybe move to when α is found?
      interpolate_weights(_vars.models[lvl-1], model, α, _vars.interpolation)

      losses[lvl]['post'] = []
      model.eval()
      for batch in _vars.batches:
        loss = fwd(_vars, model, batch, bwd=False)  # forward
        losses[lvl]['post'].append(loss.item())

      if np.mean(losses[lvl]['pre']) > np.mean(losses[lvl]['post']):
        _print(f'α = {α}')
        break
      else: 
        model = copy.deepcopy(model_copy)
        if α > 1e-2: _vars.α = α = α/2
        else: _print(f'No α has been found --> α = 0'); _vars.α = 0; break

  model.train()
  return model

def register_dl_dθ(_vars, model, lvl):
  for (p_m, g) in zip(model.parameters(), _vars.dl_dθ[lvl]):
    try: g += p_m.grad.detach()
    except: pass  # exception when a layer is unused (e.g. last one when scheme=Euler)

def reset_dl_dθ(_vars, lvl):
  for g in _vars.dl_dθ[lvl]: g.zero_()

def update_τ(_vars, lvl):
  old_model, new_model = _vars.models[lvl], _vars.models[lvl-1]
  new_optimizer = _vars.optimizers[lvl-1]

  ## Get new-grad
  for batch in _vars.batches:
    loss = fwd(_vars, new_model, batch, bwd=True)  # forward & backward
  
  torch.nn.utils.clip_grad_norm_(new_model.parameters(), .1)  # Grad. clipping

  ## new-grad to _vars.τ[lvl-1]
  for (p_nm, g) in zip(new_model.parameters(), _vars.τ[lvl-1]): 
    try: g -= p_m.grad.detach()
    except: pass  # exception when a layer is unused (e.g. last one when scheme=Euler)
    
  new_optimizer.zero_grad()

  ## Update dcorrected w/ restricted old-grad
    ## precont: emb1, emb2. 
    ## cont: encSA1-4, encFF5-8, encLN9-12
    ##       decSA1-4, decMHA5-8, decFF9-12, encLN13-18
    ## ... x N_layers!!
    ## postcont: fc1-2

  ## Pre-continuous
  for i in range(2):
    _vars.τ[lvl-1][i] += _vars.dl_dθ[lvl][i]
  n_params_enc, n_params_dec = 12, 18
  ## Continuous block
  ##  Encoder
  for i in range(_vars.models[lvl-1].encoder.N):
    for j in range(n_params_enc):
      _vars.τ[lvl-1][2 + i*n_params_enc + j] += \
                            1/2*_vars.dl_dθ[lvl][2 + (2*i)  *n_params_enc + j] \
                          + 1/2*_vars.dl_dθ[lvl][2 + (2*i+1)*n_params_enc + j]
  ##  ...copy last layer
  i = _vars.models[lvl-1].encoder.N
  for j in range(n_params_enc):
    _vars.τ[lvl-1][2 + i*n_params_enc + j] += \
                                _vars.dl_dθ[lvl][2 + (2*i)  *n_params_enc + j]        
  ##  Decoder
  transl1 = 2 + n_params_enc*(_vars.models[lvl-1].encoder.N + 1)
  transl2 = 2 + n_params_enc*(2*_vars.models[lvl-1].encoder.N + 1)
  for i in range(_vars.models[lvl-1].decoder.N):
    for j in range(n_params_dec):
      _vars.τ[lvl-1][transl1 + i*n_params_dec + j] += \
                            1/2*_vars.dl_dθ[lvl][transl2 + (2*i)  *n_params_dec + j] \
                          + 1/2*_vars.dl_dθ[lvl][transl2 + (2*i+1)*n_params_dec + j]
  ##  ...copy last layer
  i = _vars.models[lvl-1].decoder.N
  for j in range(n_params_dec):
    _vars.τ[lvl-1][transl1 + i*n_params_dec + j] += \
                                _vars.dl_dθ[lvl][transl2 + (2*i)  *n_params_dec + j]        
  ## Post-continuous
  for i in range(-2, 0):
    _vars.τ[lvl-1][i] += _vars.dl_dθ[lvl][i]

def update_dl_dθ(_vars, model, optimizer, **kwargs):
  _vars.counter_gradients_update += 1
  if _vars.counter_gradients_update % _vars.n_gradients_update == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), .1)  # Grad. clipping
    if 'register' in kwargs and kwargs['register']: 
      register_dl_dθ(_vars, model, kwargs['lvl'])
    optimizer.step()  # 6.3 Gradient accumulation
    optimizer.zero_grad()
#################################################

def V_cycle(_vars):
  t0 = time.time()
  FINE, COARSE = _vars.FINE, _vars.COARSE = _vars.n_lvls-1, 0
  assert _vars.μ % _vars.n_gradients_update == 0
  assert _vars.ν % _vars.n_gradients_update == 0
  batches = _vars.batches = [_vars.batch]#get_batches(_vars, _vars.n_gradients_update)

  ## Init V-cycle  # 0.01s
  init_dl_dθ(_vars)
  init_τ(_vars)  ## τ = d(loss correction) / d(θ)
  
  t1 = time.time(); _print(f'init V_cycle: {t1 - t0}s'); t0 = time.time()
  
  if not _vars.skip:
    ## (I) Fine --> Coarse
    losses = _vars.losses = {}
  
    for lvl in range(FINE, COARSE, -1):
      model, optimizer = _vars.models[lvl], _vars.optimizers[lvl]
      reset_dl_dθ(_vars, lvl)  # 0.0007s
      losses[lvl] = {'pre': []}

      ## Pre-smoothing  # 3s
      for k in range(_vars.μ):
        for batch in batches:
          _vars.loss = loss = fwd(_vars, model, batch, bwd=True, preds=True)  # forward & backward
          add_τ(_vars, lvl)
          if k == _vars.μ-1: losses[lvl]['pre'].append(loss.item())
          update_dl_dθ(_vars, model, optimizer, register=(k == _vars.μ-1), lvl=lvl)

      t1 = time.time(); _print(f'Pre-smoothing lvl {lvl}: {t1 - t0}s'); t0 = time.time()

      ## Restrict weights  # 0.003s
      _print(f'Restricting weights from level {lvl} to level {lvl-1}s')
      restrict_weights(model, _vars.models[lvl-1])

      t1 = time.time(); _print(f'Restrict weights: {t1 - t0}s'); t0 = time.time()

      ## Update grad-correction τ  # 0.16s
      update_τ(_vars, lvl)

      t1 = time.time(); _print(f'Update τ: {t1 - t0}s'); t0 = time.time()

    ## (II) Coarsest
    model, optimizer = _vars.models[0], _vars.optimizers[0]

    ## Smoothing at the coarsest level  # 1.6s
    for k in range(_vars.μ):
      for batch in batches:
        loss = fwd(_vars, model, batch, bwd=True)  # forward & backward
        add_τ(_vars, lvl)
        update_dl_dθ(_vars, model, optimizer)
    t1 = time.time(); _print(f'Smoothing coarsest: {t1 - t0}s'); t0 = time.time()

    ## (III) Coarse --> Fine
    for lvl in range(1, _vars.n_lvls):
      model, optimizer = _vars.models[lvl], _vars.optimizers[lvl]

      ## Line search  # 0.125s
      model = line_search(_vars, model, lvl)

      t1 = time.time(); _print(f'Line search lvl {lvl}: {t1 - t0}s'); t0 = time.time()

      ## Post-smoothing  # 3s
      ##  Modification: not fore FINE level
      if lvl != FINE:
        for k in range(_vars.ν):
          for batch in batches:
            loss = fwd(_vars, model, batch, bwd=True)  # forward & backward
            add_τ(_vars, lvl)
            update_dl_dθ(_vars, model, optimizer)

      t1 = time.time(); _print(f'Post-smoothing lvl {lvl}: {t1 - t0}s'); t0 = time.time()

  else:  # checked to output same loss as main_conv provided the same batch
         # thus, the only difference with main_conv is the repeation of μ
         #... and the random selection of the batches (related to the grads update counter)
    lvl = FINE
    model, optimizer = _vars.models[lvl], _vars.optimizers[lvl]
    reset_dl_dθ(_vars, lvl)

    ## Smoothing at the finest level  # 3s (for μ=10 and #batches=10)
    for k in range(_vars.μ):
      for batch in batches:
        _vars.loss = loss = fwd(_vars, model, batch, bwd=True, preds=True)  # forward & backward  # .03s
        update_dl_dθ(_vars, model, optimizer, register=False)  # 1e-6s

    t1 = time.time(); _print(f'Smoothing at lvl {lvl}: {t1 - t0}s'); t0 = time.time()








































