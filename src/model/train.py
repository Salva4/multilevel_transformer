import numpy as np
import os
import sys
import torch

sys.path.append('../../../src/')
from mgopt.mgopt import _MGOPT as train_mgopt
from src_utils.monitoring import time_

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
  batch_bwd_time = time_(loss.backward)

  gradient_handler.ctr += 1
  if gradient_handler.ctr%gradient_handler.size == 0:
    if gradient_handler.clipping_norm is not None:
      gradient_handler.clip_grad_norm()
    optimizer.step()
    optimizer.zero_grad()

  return batch_bwd_time

def print_times_(*times, mode):
  if mode == 'training':
    batch_fwd_time, batch_bwd_time = times
    print(f'Training batch fwd pass time: {batch_fwd_time} seconds')
    print(f'Training batch bwd pass time: {batch_bwd_time} seconds')
  elif mode == 'validation':
    batch_fwd_time = times
    print(f'Evaluation batch fwd pass time: {batch_fwd_time} seconds')
  else: raise Exception()

def update_output_loss_and_accuracy(
  output, losses, accuracy_counter, compute_accuracy,
):
  output['loss'] = np.mean(losses)
  output['accuracy'] = accuracy_counter.correct/accuracy_counter.total \
                       if compute_accuracy else None

def clean_sentence(sentence):
  pad_token, eos_token = '<pad>', '<eos>'
  sentence = sentence.replace(pad_token, '')
  first_eos = sentence.find(eos_token)
  if first_eos != -1: sentence = sentence[:first_eos + len(eos_token)]

  return sentence

def print_example_(
  model, model_inputs, split, src_decoding_function, tgt_decoding_function,
):
  model_inputs['input' ] = model_inputs['input' ][0:1]
  model_inputs['target'] = model_inputs['target'][0:1]

  model.eval()
  with torch.no_grad():
    model_outputs = model(**model_inputs)
    print()
    print(f'{split.capitalize()} example:')
    print(f'''         Model input: {clean_sentence(
      src_decoding_function(model_inputs ['input'      ][0].tolist())
    )}''')
    print(f'''    Model prediction: {clean_sentence(
      tgt_decoding_function(model_outputs['predictions'][0].tolist())
    )}''')
    print(f'''              Target: {clean_sentence(
      tgt_decoding_function(model_outputs['target'     ][0].tolist())
    )}''')
    print(f'''             Correct: {
      'Yes' if model_outputs['correct'] == 1 else 'No'
    }''')
    print()

  model.train()

def train_conventional(
  prepare_inputs, forward_pass, backward_pass, num_batches, print_example,
  print_times,
):
  for batch_idx in range(num_batches):
    model_inputs, get_batch_time = prepare_inputs()
    loss, accuracy_counter, batch_fwd_time = forward_pass(model_inputs)
    batch_bwd_time = backward_pass(loss)
    if print_times: print_times_(batch_fwd_time, batch_bwd_time, 'training')

  print_example(model_inputs=model_inputs, split='training')

def train(
  model, optimizer, device, criterion, get_batch, num_batches,
  compute_accuracy=False, print_example=False, src_decoding_function=None,
  tgt_decoding_function=None, print_times=False, use_mgopt=False,
  gradient_accumulation_size=1, gradient_clipping_norm=None, **details,
):
  output = {}
  model.train()
  optimizer.zero_grad()
  losses = []
  accuracy_counter = AccuracyCounter()
  gradient_handler = GradientHandler(
    gradient_accumulation_size, model, gradient_clipping_norm,
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
  _print_example = lambda *args, **kwargs: print_example_(
    model=model, src_decoding_function=src_decoding_function,
    tgt_decoding_function=tgt_decoding_function, *args, **kwargs,
  ) if print_example else None

  if not use_mgopt:
    train_conventional(
      prepare_inputs=_prepare_inputs, forward_pass=_forward_pass,
      backward_pass=_backward_pass, num_batches=num_batches,
      print_example=_print_example, print_times=print_times,
    )
  else:
    train_mgopt(
      model=model, optimizer=optimizer, prepare_inputs=_prepare_inputs,
      num_batches=num_batches, losses=losses, print_example=_print_example,
      **details,
    )

  update_output_loss_and_accuracy(
    output, losses, accuracy_counter, compute_accuracy,
  )

  return output

@torch.no_grad()
def evaluate(
  model, device, criterion, get_batch, num_batches, compute_accuracy=False,
  print_example=False, src_decoding_function=None, tgt_decoding_function=None,
  print_times=False, **details,
):
  output = {}
  model.eval()
  losses = []
  accuracy_counter = AccuracyCounter()

  for batch_idx in range(num_batches):
    model_inputs, get_batch_time = prepare_inputs(
      get_batch, device, criterion, compute_accuracy, details,
    )
    loss, accuracy_counter, batch_fwd_time = forward_pass(
      model, model_inputs, losses, accuracy_counter,
    )
    if print_times: print_times_(batch_fwd_time, 'validation')

  update_output_loss_and_accuracy(
    output, losses, accuracy_counter, compute_accuracy,
  )

  if print_example: print_example_(
    model, model_inputs, 'validation', src_decoding_function,
    tgt_decoding_function,
  )

  return output




