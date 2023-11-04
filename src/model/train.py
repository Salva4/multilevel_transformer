import numpy as np
import os
import sys
import torch

sys.path.append(os.path.join('..', '..', '..', 'src', 'utils'))
from monitoring import time_

class AccuracyCounter: correct = 0; total = 0

def prepare_inputs(
  get_batch, device, criterion, compute_accuracy, fwd_pass_details,
):
    batch, get_batch_time = time_(get_batch)

    input, target = batch
    input, target = input.to(device), target.to(device)

    model_inputs = {
      'input': input, 'target': target, 'criterion': criterion, 
      'compute_accuracy': compute_accuracy,
    }
    model_inputs.update(fwd_pass_details)

    return model_inputs, get_batch_time

def forward_pass(
  model, model_inputs, losses, compute_accuracy, accuracy_counter,
):
  model_outputs, batch_fwd_time = time_(model, **model_inputs)
  loss = model_outputs['loss']
  losses.append(loss.item())

  if compute_accuracy: 
    accuracy_counter.correct += model_outputs['correct']
    accuracy_counter.total += model_outputs['total']

  return loss, accuracy_counter, batch_fwd_time

def backward_pass(optimizer, loss):
  optimizer.zero_grad()#set_to_none=True) <-- ?
  batch_bwd_time = time_(loss.backward)
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

def train(
  model, optimizer, device, criterion, get_batch, num_batches, 
  compute_accuracy=False, print_times=False, **fwd_pass_details,
):
  output = {}
  model.train()
  losses = []
  accuracy_counter = AccuracyCounter()

  for batch_idx in range(num_batches):
    model_inputs, get_batch_time = prepare_inputs(
      get_batch, device, criterion, compute_accuracy, fwd_pass_details,
    )
    loss, accuracy_counter, batch_fwd_time = forward_pass(
      model, model_inputs, losses, compute_accuracy, accuracy_counter,
    )
    batch_bwd_time = backward_pass(optimizer, loss)
    if print_times: print_times_(batch_fwd_time, batch_bwd_time, 'training')

  update_output_loss_and_accuracy(
    output, losses, accuracy_counter, compute_accuracy,
  )
  return output

@torch.no_grad()
def evaluate(
  model, device, criterion, get_batch, num_batches, compute_accuracy=False, 
  print_times=False, **fwd_pass_details,
):
  output = {}
  model.eval()
  losses = []
  accuracy_counter = AccuracyCounter()

  for batch_idx in range(num_batches):
    torch.manual_seed(-(batch_idx+1))
    model_inputs, get_batch_time = prepare_inputs(
      get_batch, device, criterion, compute_accuracy, fwd_pass_details,
    )
    loss, accuracy_counter, batch_fwd_time = forward_pass(
      model, model_inputs, losses, compute_accuracy, accuracy_counter,
    )
    if print_times: print_times_(batch_fwd_time, 'evaluation')

  update_output_loss_and_accuracy(
    output, losses, accuracy_counter, compute_accuracy,
  )
  return output












