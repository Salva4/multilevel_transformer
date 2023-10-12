import math
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, _vars):
    super().__init__()
    self.fc1 = nn.Linear(_vars.d, _vars.dim_ff)
    self.fc2 = nn.Linear(_vars.dim_ff, _vars.d)
    self.activation_fn = nn.SiLU()

  def forward(self, x):
    _h = self.fc1(x)
    h = self.activation_fn(_h)
    o = self.fc2(h)
    return o

class MultiHeadAttention(nn.Module):
  def __init__(self, _vars):
    super().__init__()
    self.k_proj = nn.Linear(
      in_features=_vars.d,
      out_features=_vars.d,
      bias=True
    )
    self.v_proj = nn.Linear(
      in_features=_vars.d,
      out_features=_vars.d,
      bias=True
    )
    self.q_proj = nn.Linear(
      in_features=_vars.d,
      out_features=_vars.d,
      bias=True
    )
    self.out_proj = nn.Linear(
      in_features=_vars.d,
      out_features=_vars.d,
      bias=True
    )

  def forward(self, _K, _V, _Q):  # _K, _V, _Q: [b, L, d]
    K = self.k_proj(_K)  # K: [b, L , d]
    V = self.v_proj(_V)  # V: [b, L , d]
    Q = self.q_proj(_Q)  # Q: [b, L', d]
    KT_Q = ((K.unsqueeze(2) * Q.unsqueeze(1)).sum(-1))/np.sqrt(_vars.d)  # KT_Q: [b, L, L']
    α = KT_Q.softmax(1)  # α: [b, L, L']
    V_ = (V.unsqueeze(2) * α.unsqueeze(3)).sum(1)  # V_: [b, L', d]
    O = self.out_proj(V_)  # O: [b, L', d]
    return O

class PositionalEncoding(nn.Module): ## taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SelfAttention(nn.Module):
  def __init__(self, _vars):
    super().__init__()
    self.att = MultiHeadAttention(_vars)

  def forward(self, x):
    return self.att(x, x, x)

class Transformer(nn.Module):
  def __init__(self, _vars):
    super().__init__()
    self.embedding = nn.Embedding(
      len(_vars.tokenizer), 
      _vars.d, 
      padding_idx=_vars.pad_id
    )
    self.positional_encoding = PositionalEncoding(_vars.d)
    encoder_layer = TransformerEncoderLayer(_vars)
    decoder_layer = TransformerDecoderLayer(_vars)
    self.encoder = nn.ModuleList([
      encoder_layer for _ in range(_vars.num_layers_encoder)
    ])
    self.decoder = nn.ModuleList([
      decoder_layer for _ in range(_vars.num_layers_decoder)
    ])
    self.classifier = nn.Linear(_vars.d, len(_vars.tokenizer))

  def forward(self, src, tgt):  # src: [b, L ]
                                # tgt: [b, L']
    ## Embedding
    src = self.embedding(src)  # src: [b, L , d]
    tgt = self.embedding(tgt)  # tgt: [b, L', d]

    ## Positional encoding
    src = self.positional_encoding(src)  # src: [b, L , d]
    tgt = self.positional_encoding(tgt)  # tgt: [b, L', d]

    ## Encoder
    for layer in self.encoder:
      src = layer(src)
    mem = src  # mem: [b, L, d]

    ## Decoder
    for layer in self.decoder:
      tgt = layer(tgt, mem)  # tgt: [b, L', d]

    ## Classifier
    out = self.classifier(tgt)  # out: [b, L', m]

    return out

class TransformerDecoderLayer(nn.Module):
  def __init__(self, _vars):
    super().__init__()
    self.self_attn = SelfAttention(_vars)
    self.cross_attn = MultiHeadAttention(_vars)
    self.mlp = MLP(_vars)
    self.self_attn_layer_norm = nn.LayerNorm(
      (_vars.d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.cross_attn_layer_norm = nn.LayerNorm(
      (_vars.d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.final_layer_norm = nn.LayerNorm(
      (_vars.d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.norm_first = _vars.norm_first

  def forward(self, x, memory):
    if self.norm_first:
      x = self.self_attn(self.self_attn_layer_norm(x))
      x = self.cross_attn(
        _K=memory, 
        _V=memory, 
        _Q=self.cross_attn_layer_norm(x)
      )
      x = self.mlp(self.final_layer_norm(x))

    else: 
      x = self.self_attn_layer_norm(self.self_attn(x))
      x = self.cross_attn_layer_norm(self.cross_attn(x))
      x = self.final_layer_norm(self.mlp(x))

    return x

class TransformerEncoderLayer(nn.Module):
  def __init__(self, _vars):
    super().__init__()
    self.self_attn = SelfAttention(_vars)
    self.mlp = MLP(_vars)
    self.self_attn_layer_norm = nn.LayerNorm(
      (_vars.d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.final_layer_norm = nn.LayerNorm(
      (_vars.d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.norm_first = _vars.norm_first

  def forward(self, x):
    if self.norm_first:
      x = self.self_attn(self.self_attn_layer_norm(x))
      x = self.mlp(self.final_layer_norm(x))

    else: 
      x = self.self_attn_layer_norm(self.self_attn(x))
      x = self.final_layer_norm(self.mlp(x))

    return x

























