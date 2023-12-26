import torch.nn as nn

# from .methods.init_weights import init_weights

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




