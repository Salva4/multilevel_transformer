import numpy as np
import torch.nn as nn

class MultiHeadAttention(nn.Module):
  def __init__(self, model_dimension, num_heads):
    super().__init__()

    self.model_dimension = model_dimension
    self.num_heads = num_heads
    self.dim_keys = self.model_dimension//self.num_heads
    self.dim_values = self.dim_keys

    self.k_proj = nn.Linear(
      in_features=self.model_dimension,
      out_features=self.num_heads*self.dim_keys,
      bias=True,
    )
    self.v_proj = nn.Linear(
      in_features=self.model_dimension,
      out_features=self.num_heads*self.dim_values,
      bias=True,
    )
    self.q_proj = nn.Linear(
      in_features=self.model_dimension,
      out_features=self.num_heads*self.dim_keys,
      bias=True,
    )
    self.out_proj = nn.Linear(
      in_features=self.num_heads*self.dim_values,
      out_features=self.model_dimension,
      bias=True,
    )

  def forward(self, _K, _V, _Q, mask_attn=None, mask_pad=None):  
    '''
        _K, _V: [b , L , d]
            _Q: [b , L', d]
     mask_attn: [L', L ]
     mask_pad : [b , L ]
    '''
    b, L, d, Lp = *_K.shape, _Q.shape[1]

    # print(f'_K.shape {_K.shape}')
    # print(f'_V.shape {_V.shape}')
    # print(f'_Q.shape {_Q.shape}')
    # if mask_attn is not None: print(f'mask_attn {mask_attn.shape}')
    # if mask_pad  is not None: print(f'mask_pad  {mask_pad .shape}')
    # print(f'_K _V _Q {_K.ravel()[:4]} {_V.ravel()[:4]} {_Q.ravel()[:4]}')

    assert _K.shape == (b, L , d)
    assert _V.shape == (b, L , d)
    assert _Q.shape == (b, Lp, d)
    if mask_attn is not None: assert mask_attn.shape == (Lp, L), f'mask_attn.shape {mask_attn.shape}, Lp {Lp}, L {L}'
    if mask_pad  is not None: assert mask_pad .shape == (b , L), f'mask_pad.shape { mask_pad .shape}, b {  b}, L {L}'
    nh, dk, dv = self.num_heads, self.dim_keys, self.dim_values

    K = self.k_proj(_K).reshape(b, L , nh, dk).transpose(1, 2)  # K: [b, nh, L , dk]
    V = self.v_proj(_V).reshape(b, L , nh, dv).transpose(1, 2)  # V: [b, nh, L , dv]
    Q = self.q_proj(_Q).reshape(b, Lp, nh, dk).transpose(1, 2)  # Q: [b, nh, L', dk]

    # print(f'K V Q {K.ravel()[:4]} {V.ravel()[:4]} {Q.ravel()[:4]}')

    KQ = (K @ Q.transpose(-2, -1)) / np.sqrt(dk)  # KQ: [b, nh, L, L']

    if mask_pad is not None:
      mask_pad = mask_pad.reshape(b, 1, L, 1)  # mask_pad: [b, 1, L, 1]
      KQ += mask_pad  # KQ: [b, nh, L, L']

    if mask_attn is not None:  # only in self-attention --> L=L'.
      mask_attn = mask_attn.reshape(1, 1, Lp, L)  # mask_attn: [1, 1, L', L]
      KQ += mask_attn.transpose(-2, -1)  # KQ: [b, nh, L, L']

    α = KQ.softmax(2)  # α: [b, nh, L, L']
    O = (α.transpose(-2, -1) @ V)  # O: [b, nh, L', dv]
    O = O.transpose(1, 2).reshape(b, Lp, nh*dv)  # O: [b, L', nh·dv]
    out = self.out_proj(O)  # out: [b, L', d]

    # print(f'α O out {α.ravel()[:4]} {O.ravel()[:4]} {out.ravel()[:4]}')

    return out




