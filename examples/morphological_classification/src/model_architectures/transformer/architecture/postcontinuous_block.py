import torch.nn as nn

class PostContinuousBlock(nn.Module):
  def __init__(self, model_dimension, num_classes, **kwargs):
    super().__init__()#**kwargs)
    self.d = model_dimension
    self.m = num_classes

    self.fc3 = nn.Linear(self.d, self.m)
    self.ln3 = nn.LayerNorm(self.d)

  def forward(self, x, **kwargs):
    x = self.ln3(x)
    x = self.fc3(x)

    return {'x': x}




