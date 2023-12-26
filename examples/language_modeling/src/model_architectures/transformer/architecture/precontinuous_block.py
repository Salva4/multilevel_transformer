import torch
import torch.nn as nn

# from .methods.init_weights import init_weights

class PreContinuousBlock(nn.Module):
  def __init__(
    self, model_dimension, vocabulary_size, context_window, **kwargs
  ):
    super().__init__()
    self.emb = nn.Embedding(vocabulary_size, model_dimension)
    self.posenc = nn.Embedding(context_window, model_dimension)
    # self.apply(init_weights)

  def forward(self, x, **kwargs):
    B, T = x.shape
    positions = torch.arange(T, device=x.device)
    x = self.emb(x) + self.posenc(positions)  # (B,T,C) + (T,C) = (B,T,C)

    return {'x': x}




