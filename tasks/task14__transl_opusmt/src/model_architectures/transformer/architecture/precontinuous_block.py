import torch
import torch.nn as nn

class PreContinuousBlock(nn.Module):
  def __init__(self, model_dimension, tokenizer, device, pad_id, **kwargs):
    super().__init__()

    self.model_dimension = model_dimension  # aka 'd'
    self.device = device
    self.pad_id = pad_id

    self.embedding = nn.Embedding(
      len(tokenizer),
      model_dimension,
      padding_idx=pad_id,
    )

    self.positional_encoding_src = nn.Embedding(512, 512)
    self.positional_encoding_tgt = nn.Embedding(512, 512)

    # self.apply(init_weights)

  def forward(self, x, y, **kwargs):  # x: [b, L   ]
                                      # y: [b, L'+1]
    src = x          # src: [b, L ]
    tgt = y[:, :-1]  # tgt: [b, L']
    labels = y[:, 1:]

    ## Padding masks for attention
    mask_pad_src = torch.where(src.eq(self.pad_id), -np.inf, 0)  # mask_pad_src: [b, L ]
    mask_pad_tgt = torch.where(tgt.eq(self.pad_id), -np.inf, 0)  # mask_pad_tgt: [b, L']
    mask_pad_mem = mask_pad_src                                  # mask_pad_mem: [b, L ]

    ## Embedding
    x = self.embedding(src)  # src: [b, L , d]
    y = self.embedding(tgt)  # tgt: [b, L', d]

    ## Scale
    x *= np.sqrt(self.model_dimension)
    y *= np.sqrt(self.model_dimension)

    ## Positional encoding
    L  = x.shape[1]
    Lp = y.shape[1]
    positions_src = torch.arange(L ).reshape(1, L ).to(self.device)  # positions_src: [1, L ]
    positions_tgt = torch.arange(Lp).reshape(1, Lp).to(self.device)  # positions_tgt: [1, L']
    positional_encoding_src = self.positional_encoding_src(positions_src)  # positions_src: [1, L , d]
    positional_encoding_tgt = self.positional_encoding_tgt(positions_tgt)  # positions_tgt: [1, L', d]

    x += positional_encoding_src  # src: [b, L , d]
    y += positional_encoding_tgt  # tgt: [b, L', d]

    return {
      'x': x,
      'y': y,
      'mask_pad_src': mask_pad_src,
      'mask_pad_tgt': mask_pad_tgt,
      'mask_pad_mem': mask_pad_mem,
      'target': labels,
    }




