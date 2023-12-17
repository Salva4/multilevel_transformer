import torch.nn as nn

class PreContinuousBlock(nn.Module):
  def __init__(self, vocabulary_size, model_dimension, **kwargs):
    super().__init__()
    self.emb = nn.Embedding(vocabulary_size, model_dimension)#(15514, 256)

  def forward(self, x, **kwargs):
    x = self.emb(x)

    return {'x': x}



