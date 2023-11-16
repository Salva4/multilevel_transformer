## Adapted from transformers/generation/utils.py (HuggingFace)

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

# from transformers.pytorch_utils import torch_int_div
from transformers.utils import ModelOutput, logging
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.generation.logits_process import (
  TopKLogitsWarper,
  TopPLogitsWarper,
)

@torch.no_grad()
def generate(model, src, do_sample, num_beams=4, **kwargs):  # src: [b, L]
  if do_sample: raise Exception('Not implemented yet.')

  batch_size = src.shape[0]
  pad_token_id = kwargs['pad_token_id']  # 58100
  src_embedding_fn = model.precontinuous_block.embed_src
  encoder = model.continuous_blocks[0]

  state = {'x': src}
  state.update(src_embedding_fn(**state))
  state.update(encoder     (**state))
  memory = state['x']
  mask_pad_src = state['mask_pad_src']
  mask_pad_mem = state['mask_pad_mem']

  generated_sequence = torch.empty((batch_size, 1), device=memory.device) \
                                  .fill_(pad_token_id).long()
  beam_scorer = BeamSearchScorer(
    batch_size=batch_size,
    num_beams=num_beams,
    device=memory.device,
    length_penalty=1.,
    do_early_stopping=False,
    num_beam_hyps_to_keep=1,
  )

  memory = memory.repeat_interleave(num_beams, dim=0)
  mask_pad_src = mask_pad_src.repeat_interleave(num_beams, dim=0)
  mask_pad_mem = mask_pad_mem.repeat_interleave(num_beams, dim=0)
  generated_sequence = generated_sequence.repeat_interleave(num_beams, dim=0)

  state['memory'] = memory
  state['mask_pad_src'] = mask_pad_src
  state['mask_pad_mem'] = mask_pad_mem
  state['tgt_input'] = generated_sequence
  state['mask_pad_tgt'] = None

  return beam_search(
    model,
    state,
    beam_scorer,
    **kwargs,
  )

def beam_search(
  model, state, beam_scorer, max_new_tokens, pad_token_id, eos_token_id, 
  **kwargs,
):
  batch_size = len(beam_scorer._beam_hyps)
  num_beams = beam_scorer.num_beams

  batch_beam_size, current_length = state['tgt_input'].shape

  if num_beams * batch_size != batch_beam_size: raise ValueError(f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.")

  # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
  # of the first beam are considered to avoid sampling the exact same tokens across all beams.
  beam_scores = torch.zeros(
    (batch_size, num_beams), 
    dtype=torch.float, 
    device=state['tgt_input'].device,
  )
  beam_scores[:, 1:] = -1e9
  beam_scores = beam_scores.view((batch_size * num_beams,))

  tgt_embedding_fn = model.precontinuous_block.embed_tgt
  decoder = model.continuous_blocks[1]
  classifier = model.postcontinuous_block

  while True:
    memory, generated_sequence = state['memory'], state['tgt_input']
    state['x'], state['y'] = memory, generated_sequence
    state['y'] = tgt_embedding_fn(**state, split_target=False)['y']
    state.update(decoder         (**state))
    state.update(classifier      (**state))
    logits = state['x']

    next_token_logits = logits[:, -1, :]
    next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

    next_token_scores += beam_scores[:, None].expand_as(next_token_scores)

    # reshape for beam search
    vocab_size = next_token_scores.shape[-1]
    next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

    # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
    next_token_scores, next_tokens = torch.topk(
      next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
    )

    next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')#torch_int_div(next_tokens, vocab_size)
    next_tokens = next_tokens % vocab_size

    # stateless
    beam_outputs = beam_scorer.process(
      generated_sequence,
      next_token_scores,
      next_tokens,
      next_indices,
      pad_token_id=pad_token_id,
      eos_token_id=eos_token_id,
      beam_indices=None,
    )

    beam_scores = beam_outputs['next_beam_scores']
    beam_next_tokens = beam_outputs['next_beam_tokens']
    beam_idx = beam_outputs['next_beam_indices']

    generated_sequence = torch.cat(
      [generated_sequence[beam_idx, :], beam_next_tokens.unsqueeze(-1)], 
      dim=-1,
    )
    state['tgt_input'] = generated_sequence

    current_length += 1

    if beam_scorer.is_done or current_length == max_new_tokens + 1: break

  sequence_outputs = beam_scorer.finalize(
    generated_sequence,
    beam_scores,
    next_tokens,
    next_indices,
    pad_token_id=pad_token_id,
    eos_token_id=eos_token_id,
    max_length=max_new_tokens + 1,
    beam_indices=None,
  )
  return sequence_outputs['sequences']

# def top_k_top_p_filtering(
#   logits: torch.FloatTensor,
#   top_k: int = 0,
#   top_p: float = 1.0,
#   filter_value: float = -float("Inf"),
#   min_tokens_to_keep: int = 1,
# ):
#   if top_k > 0:
#     logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
#       None, logits
#     )

#   if 0 <= top_p <= 1.0:
#     logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
#       None, logits
#     )

#   return logits




