import torch
import torch.nn as nn
import torch.nn.functional as F

from .methods.init_weights import init_weights
from ._utils.F_dec import F_dec

class ContinuousResidualLayer(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.F = F_dec(**kwargs)
    self.apply(init_weights)

  def forward(self, x, **kwargs): return {'x': self.F(x)}

class PostContinuousBlock(nn.Module):
  def __init__(self, d_model, vocab_size, **kwargs):
    super().__init__()
    self.ln = nn.LayerNorm(d_model) # final layer norm
    self.classifier = nn.Linear(d_model, vocab_size)
    self.apply(init_weights)

  def forward(self, x, **kwargs):
    x = self.ln(x) # (B,T,C)
    logits = self.classifier(x) # (B,T,vocab_size)

    loss = F.cross_entropy(
      logits.view(-1, logits.shape[-1]), 
      kwargs['targets'].view(-1),
    ) if 'targets' in kwargs else None

    return {'x': logits, 'loss': loss}

class PreContinuousBlock(nn.Module):
  def __init__(self, d_model, vocab_size, block_size, **kwargs):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, d_model)
    self.posenc = nn.Embedding(block_size, d_model)
    self.apply(init_weights)

  def forward(self, x, **kwargs):
    B, T = x.shape
    positions = torch.arange(T, device=x.device)
    x = self.emb(x) + self.posenc(positions)  # (B,T,C) + (T,C) = (B,T,C)

    return {'x': x}



