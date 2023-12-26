import numpy as np
import torch
import torch.nn as nn

from ..model_utils.marian_sinusoidal_positional_embedding \
  import MarianSinusoidalPositionalEmbedding

class PreContinuousBlock(nn.Module):
  def __init__(
    self, model_dimension, tokenizer, device, pad_token_id, **kwargs
  ):
    super().__init__()

    self.model_dimension = model_dimension  # aka 'd'
    self.device = device
    self.pad_token_id = pad_token_id

    self.embedding = nn.Embedding(
      len(tokenizer),
      model_dimension,
      padding_idx=pad_token_id,
    )

    # ## set to sinusoidal and no-grad --> later?
    # self.positional_encoding_src = nn.Embedding(512, model_dimension)
    # self.positional_encoding_tgt = nn.Embedding(512, model_dimension)
    self.positional_encoding_src = MarianSinusoidalPositionalEmbedding(
      512, model_dimension, pad_token_id,
    )
    self.positional_encoding_tgt = MarianSinusoidalPositionalEmbedding(
      512, model_dimension, pad_token_id,
    )

    # self.apply(init_weights) --> do like MarianModel

  def forward(self, **state):
    ## Debugging enc-dec gradient_function
    # return {'mask_pad_src': torch.zeros(size=state['x'].shape[:2], device=state['x'].device), 
    #         'mask_pad_mem': torch.zeros(size=state['x'].shape[:2], device=state['x'].device), 
    #         'mask_pad_tgt': torch.zeros(size=state['y'].shape[:2], device=state['x'].device), }  # debug

    state_update = {}
    state_update.update(self.embed_src(**state))
    state_update.update(self.embed_tgt(**state))
    return state_update

  def embed_src(self, x, **kwargs):  # x: [b, L   ]
    src = x  # src: [b, L ]

    ## Padding masks for attention
    mask_pad_src = torch.where(src.eq(self.pad_token_id), -np.inf, 0) \
                   .to(src.device)  # mask_pad_src: [b, L]
    mask_pad_mem = mask_pad_src     # mask_pad_mem: [b, L]

    ## Embedding
    x = self.embedding(src)  # src: [b, L , d]

    ## Scaling
    x *= np.sqrt(self.model_dimension)

    ## Positional encoding
    # L  = x.shape[1]
    # positions_src = torch.arange(L).reshape(1, L).to(self.device)  # positions_src: [1, L ]
    # positional_encoding_src = self.positional_encoding_src(positions_src)  # positions_src: [1, L , d]
    positional_encoding_src = self.positional_encoding_src(src.shape)

    x += positional_encoding_src  # src: [b, L , d]

    # print(f'x {x.ravel()[:10]}')
    # import sys; sys.exit()

    return {
      'x': x, 
      'mask_pad_src': mask_pad_src, 
      'mask_pad_mem': mask_pad_mem,
    }

  def embed_tgt(self, y, split_target=True, **kwargs):  # y: [b, L'+1]
    '''split_target is True during a conventional forward pass, where the 
    target must be split into target_inputs (to the model) and labels.
    However, during generation, the targe_inputs are the whole target tensor,
    so split_arget is False.'''

    if split_target: 
      tgt = y[:, :-1]    #    tgt: [b, L']
      labels = y[:, 1:]  # labels: [b, L']
    else:
      tgt = y
      labels = None

    ## Padding masks for attention
    mask_pad_tgt = torch.where(tgt.eq(self.pad_token_id), -np.inf, 0) \
                   .to(tgt.device)  # mask_pad_tgt: [b, L']

    ## Embedding
    y = self.embedding(tgt)  # tgt: [b, L', d]

    ## Scaling
    y *= np.sqrt(self.model_dimension)

    ## Positional encoding
    # Lp = y.shape[1]
    # positions_tgt = torch.arange(Lp).reshape(1, Lp).to(self.device)  # positions_tgt: [1, L']
    # positional_encoding_tgt = self.positional_encoding_tgt(positions_tgt)  # positions_tgt: [1, L', d]
    positional_encoding_tgt = self.positional_encoding_src(tgt.shape)

    y += positional_encoding_tgt  # tgt: [b, L', d]

    # print(f'y {y.ravel()[:5]}')
    # import sys; sys.exit()

    return {
      'y': y,
      'mask_pad_tgt': mask_pad_tgt,
      'target': labels,
    }




