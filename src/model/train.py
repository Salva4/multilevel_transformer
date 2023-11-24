import numpy as np
import os
import sys
import torch

sys.path.append('../../../src/')
from mgopt.mgopt import _MGOPT as train_mgopt
from src_utils.monitoring import time_

# sys.path.append(os.path.join('..', 'continuous_model'))
# from continuous_model import ContinuousModel

class AccuracyCounter: correct = 0; total = 0

class GradientHandler: 
  ctr = 0
  def __init__(self, size, model, clipping_norm=None):
    self.size = size
    self.model = model
    self.clipping_norm = clipping_norm

  def clip_grad_norm(self):
    torch.nn.utils.clip_grad_norm_(
      self.model.parameters(), max_norm=self.clipping_norm,
    )

def prepare_inputs(
  get_batch, device, criterion, compute_accuracy, details,
):
    batch, get_batch_time = time_(get_batch)

    input, target = batch
    input, target = input.to(device), target.to(device)

    model_inputs = {
      'input': input, 'target': target, 'criterion': criterion,
      'compute_accuracy': compute_accuracy,
    }
    model_inputs.update(details)

    return model_inputs, get_batch_time

def forward_pass(model, model_inputs, losses, accuracy_counter):
  model_outputs, batch_fwd_time = time_(model, **model_inputs)
  loss = model_outputs['loss']
  losses.append(loss.item())

  if model_inputs['compute_accuracy']:
    accuracy_counter.correct += model_outputs['correct']
    accuracy_counter.total += model_outputs['total']

  return loss, accuracy_counter, batch_fwd_time

def backward_pass(optimizer, loss, gradient_handler):
  optimizer.zero_grad()#set_to_none=True) <-- ?
  batch_bwd_time = time_(loss.backward)
  
  gradient_handler.ctr += 1
  if gradient_handler.ctr%gradient_handler.size == 0:
    if gradient_handler.clipping_norm is not None:
      gradient_handler.clip_grad_norm()
    optimizer.step()

  return batch_bwd_time

def print_times_(*times, mode):
  if mode == 'training':
    batch_fwd_time, batch_bwd_time = times
    print(f'Training batch fwd pass time: {batch_fwd_time} seconds')
    print(f'Training batch bwd pass time: {batch_bwd_time} seconds')
  elif mode == 'evaluation':
    batch_fwd_time = times
    print(f'Evaluation batch fwd pass time: {batch_fwd_time} seconds')
  else: raise Exception()

def update_output_loss_and_accuracy(
  output, losses, accuracy_counter, compute_accuracy,
):
  output['loss'] = np.mean(losses)
  output['accuracy'] = accuracy_counter.correct/accuracy_counter.total \
                       if compute_accuracy else None

def train_conventional(
  prepare_inputs, forward_pass, backward_pass, num_batches, print_times,
):
  for batch_idx in range(num_batches):
    model_inputs, get_batch_time = prepare_inputs()
    loss, accuracy_counter, batch_fwd_time = forward_pass(model_inputs)
    batch_bwd_time = backward_pass(loss)
    if print_times: print_times_(batch_fwd_time, batch_bwd_time, 'training')

def train(
  model, optimizer, device, criterion, get_batch, num_batches,
  compute_accuracy=False, print_times=False, use_mgopt=False, 
  gradient_accumulation_size=1, clipping_norm=None, **details,
):
  # if use_mgopt: assert isinstance(model, ContinuousModel)

  output = {}
  model.train()
  losses = []
  accuracy_counter = AccuracyCounter()
  gradient_handler = GradientHandler(
    gradient_accumulation_size, model, clipping_norm,
  )

  _prepare_inputs = lambda: prepare_inputs(
    get_batch, device, criterion, compute_accuracy, details,
  )
  _forward_pass = lambda model_inputs: forward_pass(
    model, model_inputs, losses, accuracy_counter,
  )
  _backward_pass = lambda loss: backward_pass(
    optimizer, loss, gradient_handler,
  )

  if not use_mgopt:
    train_conventional(
      prepare_inputs=_prepare_inputs, forward_pass=_forward_pass,
      backward_pass=_backward_pass, num_batches=num_batches,
      print_times=print_times,
    )
  else:
    train_mgopt(
      model=model, optimizer=optimizer, prepare_inputs=_prepare_inputs,
      num_batches=num_batches, losses=losses, **details,
    )

  update_output_loss_and_accuracy(
    output, losses, accuracy_counter, compute_accuracy,
  )
  return output

@torch.no_grad()
def evaluate(
  model, device, criterion, get_batch, num_batches, compute_accuracy=False,
  print_times=False, **details,
):
  output = {}
  model.eval()
  losses = []
  accuracy_counter = AccuracyCounter()

  for batch_idx in range(num_batches):
    torch.manual_seed(-(batch_idx+1))
    model_inputs, get_batch_time = prepare_inputs(
      get_batch, device, criterion, compute_accuracy, details,
    )
    loss, accuracy_counter, batch_fwd_time = forward_pass(
      model, model_inputs, losses, accuracy_counter,
    )
    if print_times: print_times_(batch_fwd_time, 'evaluation')

  update_output_loss_and_accuracy(
    output, losses, accuracy_counter, compute_accuracy,
  )
  return output


