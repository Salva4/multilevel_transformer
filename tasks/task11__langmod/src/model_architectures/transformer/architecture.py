import torch
import torch.nn as nn
import torch.nn.functional as F

# from .methods.init_weights import init_weights
from ._utils.F_dec import F_dec

class PreContinuousBlock(nn.Module):
  def __init__(self, model_dimension, vocabulary_size, context_window, **kwargs):
    super().__init__()
    self.emb = nn.Embedding(vocabulary_size, model_dimension)
    self.posenc = nn.Embedding(context_window, model_dimension)
    # self.apply(init_weights)

  def forward(self, x, **kwargs):
    B, T = x.shape
    positions = torch.arange(T, device=x.device)
    x = self.emb(x) + self.posenc(positions)  # (B,T,C) + (T,C) = (B,T,C)

    return {'x': x}

class ContinuousResidualLayer(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.F = F_dec(**kwargs)
    # self.apply(init_weights)

  def forward(self, x, **kwargs): #return {'x': self.F(x)}
    x = self.F(x)
    return {'x': x}

class PostContinuousBlock(nn.Module):
  def __init__(self, model_dimension, vocabulary_size, **kwargs):
    super().__init__()
    self.ln = nn.LayerNorm(model_dimension) # final layer norm
    self.classifier = nn.Linear(model_dimension, vocabulary_size)
    # self.apply(init_weights)

  def forward(self, x, **kwargs):
    x = self.ln(x) # (B,T,C)
    x = self.classifier(x) # (B,T,vocabulary_size)

    return {'x': x}


