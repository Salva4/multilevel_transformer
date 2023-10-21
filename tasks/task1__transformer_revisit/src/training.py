import numpy as np
import torch
import sys

# from input_pipeline import PAD_ID
# import MGOPT_NN

# import tqdm

## Colored output in terminal
# try:
#   from termcolor import colored
# except:
#   color = lambda z, col: print(z)
# else:
#   color = lambda z, col: print(colored(z), col)
color = lambda z, col: print(z)

############### ML Weight initialization
# def train(
#   train_dl,
#   eval_dl,
#   model, 
#   optimizer, 
#   criterion,
#   device,
#   num_epochs,
#   n_monitoring,
# ):

#   ## Early stop
#   # maxVaAcc = -1.
#   # patience_ctr = 0 
#   # patience = 200

#   batch_epochs = num_epochs   # 1 epoch is too large
#   batch_ctr = 0

#   model.train()
#   for i, batch in enumerate(train_dl):
#     inputs, targets = batch
#     # inputs, targets = inputs.long(), targets.long()
#     inputs, targets = inputs.to(device), targets.to(device)
#     # inputs = inputs.to(device) # only inputs, no targets 1/2

#     outputs = model(inputs)#.cpu() 2/2
#     loss = criterion(
#       outputs.reshape(-1, outputs.shape[-1]), 
#       targets.reshape(-1)
#     )
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if i%n_monitoring == 0:
#       model.eval()
#       with torch.no_grad():
#         ## Training accuracy
#         predictions = outputs.argmax(axis=-1)
#         tr_accuracy = ((predictions == targets)*(targets != PAD_ID)
#           ).sum()/(targets != PAD_ID).sum()

#         ## Validation accuracy
#         eval_iter = iter(eval_dl)
#         batch = next(eval_iter, None)
#         va_accuracy = []
#         while batch != None:
#           inputs, targets = batch   # both inputs & targets --> long
#           # inputs, targets = inputs.long(), targets.long()
#           inputs, targets = inputs.to(device), targets.to(device)

#           outputs = model(inputs)
#           predictions = outputs.argmax(axis=-1)
#           va_accuracy.append(
#             (
#               ((predictions == targets)*(targets != PAD_ID)).sum()/(targets != PAD_ID).sum()
#             ).item()
#           )

#           batch = next(eval_iter, None)

#         ## Validation data: 5x187 + 182
#         va_accuracy = np.mean(va_accuracy)

#         print(f'(Batch-)epoch {str(batch_ctr).zfill(2)}\tTr/Va:\t{tr_accuracy*100 : .2f}%\t{va_accuracy*100 : .2f}%')
      
#       ## Early stop
#       # if va_accuracy > maxVaAcc:
#       #   patience_ctr = 0
#       #   maxVaAcc = va_accuracy

#       # else:
#       #   patience_ctr += 1
#       #   if patience_ctr > patience:
#       #     break

#       model.train()

#     batch_ctr += 1
#     if batch_ctr == batch_epochs:
#       break
#     elif batch_ctr%(num_epochs//4) == 0:
#       for g in optimizer.param_groups:
#         g['lr'] *= .5
#       print('change lr *= .5')

#   return model
########################################

# def monitor_accs(eval_dl, model, device, inputs_tr, targets_tr, batch_ctr):
#   model.eval()

#   with torch.no_grad():
#     ## Training accuracy
#     outputs_tr = model(inputs_tr)
#     predictions_tr = outputs_tr.argmax(axis=-1)
#     tr_accuracy = (
#       (predictions_tr == targets_tr)*(targets_tr != PAD_ID)
#     ).sum() / (targets_tr != PAD_ID).sum()

#     ## Validation accuracy
#     eval_iter = iter(eval_dl)
#     batch = next(eval_iter, None)
#     va_correct, va_total = 0, 0
#     while batch != None:
#       inputs_va, targets_va = batch   # both inputs & targets --> long
#       # inputs_va, targets_va = inputs_va.long(), targets_va.long()
#       inputs_va, targets_va = inputs_va.to(device), targets_va.to(device)

#       outputs_va = model(inputs_va)
#       predictions_va = outputs_va.argmax(axis=-1)
#       va_correct += ((predictions_va == targets_va)*(targets_va != PAD_ID)).sum().item()
#       va_total += (targets_va != PAD_ID).sum().item()

#       batch = next(eval_iter, None)

#     ## Validation data: 5x187 + 182 <-- irrelevant now, acc well computed
#     va_accuracy = va_correct/va_total

#     color(f'\t(Batch-)epoch {str(batch_ctr).zfill(2)}\tTr/Va:\t{tr_accuracy*100 : .2f}%\t{va_accuracy*100 : .2f}%',
#       'yellow')
  
#   ## Early stop
#   # if va_accuracy > maxVaAcc:
#   #   patience_ctr = 0
#   #   maxVaAcc = va_accuracy

#   # else:
#   #   patience_ctr += 1
#   #   if patience_ctr > patience:
#   #     break

#   model.train()


################################# MG/OPT
# def train_MGOPT(
#   train_dl,
#   eval_dl,
#   models, 
#   optimizers, 
#   criterion,
#   device,
#   #num_epochs,
#   n_monitoring,
#   n_V_cycles,
#   mus_nus,
#   lr_MGOPT,
#   pr=False,
# ):
#   # batch_epochs = num_epochs   # 1 epoch is too large
#   batch_ctr = 0
# 
#   ## Preparation
#   for model in models:
#     model.train()
# 
#   laux = mus_nus.strip().split('_')
#   assert len(laux) == len(models)
#   mus_nus = []
#   for i, mu_nu in enumerate(laux):
#     if i == 0:
#       mu = int(mu_nu)
#       mus_nus.append(mu)
# 
#     else:
#       mu, nu = (int(x) for x in mu_nu.strip().split('-'))
#       mus_nus.append((mu, nu))
# 
#   loss_fn = lambda outputs, targets: criterion(outputs.reshape(-1, outputs.shape[-1]), 
#                            targets.reshape(-1))
# 
#   # MGOPT_NN.initIR(models)       <-- not necessary bc iterpolation & restriction are performed w/o matrix mult
#   ## TODO: optimize ? by interp/restr weights by matrix mult instead of 1 by 1
# 
#   avoid_MGOPT = abs(lr_MGOPT - (-1)) < 1e-5     # if lr_MGOPT == -1., then avoid MG_OPT
# 
#   ## Execution
#   if not avoid_MGOPT:
#     print('Performing MGOPT')
#     for i, batch in tqdm.tqdm(enumerate(train_dl)):
#       inputs, targets = batch
#       # inputs, targets = inputs.long(), targets.long()
#       inputs, targets = inputs.to(device), targets.to(device)#targets?
#       # inputs = inputs.to(device) # only inputs, no targets 1/2
# 
#       for j in range(n_V_cycles):
#         if pr: color(f'\tV-cycle #{j}', 'yellow')
#         MGOPT_NN.V_cycle(models, inputs, targets, optimizers, loss_fn, mus_nus, lr_MGOPT)
# 
#       model = models[-1]
# 
#       if i%n_monitoring == 0:
#         ## Monitoring
#         monitor_accs(eval_dl, model, device, inputs, targets, batch_ctr)
# 
#       batch_ctr += 1
#       # if batch_ctr == batch_epochs:
#       #   break
# 
#       # if ...
#         # for g in optimizer.param_groups:
#         #   g['lr'] *= .5
#         # print('change lr *= .5')
#   else:
#     print('Avoiding MGOPT')
#     model = models[-1]
#     optimizer = optimizers[-1]
#     model.train()
#     for i, batch in tqdm.tqdm(enumerate(train_dl)):
#       inputs, targets = batch
#       # inputs, targets = inputs.long(), targets.long()
#       inputs, targets = inputs.to(device), targets.to(device)#targets?
#       # inputs = inputs.to(device) # only inputs, no targets 1/2
# 
#       outputs = model(inputs)
#       loss = loss_fn(outputs, targets)
#       optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()
# 
#       model = models[-1]
# 
#       if i%n_monitoring == 0:
#         ## Monitoring
#         monitor_accs(eval_dl, model, device, inputs, targets, batch_ctr)
# 
#       batch_ctr += 1
#       # if batch_ctr == batch_epochs:
#       #   break
# 
#       # if ...
#         # for g in optimizer.param_groups:
#         #   g['lr'] *= .5
#         # print('change lr *= .5')
# 
#   return model
########################################


############### Conventional training
def train_epoch(
  train_dl, eval_dl, model, optimizer, criterion, device, level, mgrit,
):
  ## Training
  model.train()
  losses = np.empty(shape=(len(train_dl)))
  correct, total = 0, 0
  for i, batch in enumerate(train_dl):
    input_ids, target_ids = batch
    # inputs, targets = inputs.long(), targets.long()
    input_ids, target_ids = input_ids.to(device), target_ids.to(device)
    # inputs = inputs.to(device) # only inputs, no targets 1/2

    model_inputs = {
      'input': input_ids, 'target': target_ids, 'criterion': criterion, 
      'compute_accuracy': True,
    }

    if not mgrit: model_inputs.update({'level': level})
    else: 
      model_inputs.update({
      'MGRIT': True, 'relaxation': 'F', 'num_levels': 2, 'num_iterations': 2
    })

    outputs = model(**model_inputs)#.cpu() 2/2
    # print(outputs.ravel()[-10])
    loss = outputs['loss']
    losses[i] = loss.item()
    correct += outputs['correct']
    total += outputs['total']

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  training_loss = losses.mean()
  training_accuracy = correct/total

  ## Evaluation
  model.eval()
  with torch.no_grad():
    losses = np.empty(shape=(len(eval_dl)))
    correct, total = 0, 0
    for i, batch in enumerate(eval_dl):
      input_ids, target_ids = batch
      input_ids, target_ids = input_ids.to(device), target_ids.to(device)

      model_inputs = {
        'input': input_ids, 'target': target_ids, 'criterion': criterion, 
        'compute_accuracy': True,
      }

      if not mgrit: model_inputs.update({'level': level})
      else: 
        model_inputs.update({
        'MGRIT': True, 'relaxation': 'F', 'num_levels': 2, 'num_iterations': 2
      })

      outputs = model(**model_inputs)
      loss = outputs['loss']
      losses[i] = loss.item()
      correct += outputs['correct']
      total += outputs['total']

  ## Validation data: 5x187 + 182
  validation_loss = losses.mean()
  validation_accuracy = correct/total
    
  return (
    model, training_loss, training_accuracy, validation_loss, 
    validation_accuracy,
  )
########################################

































