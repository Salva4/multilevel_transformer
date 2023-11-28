import torch.nn as nn

class ContinuousResidualLayer(nn.Module):
  def __init__(self, model_dimension, **kwargs):
    super().__init__()
    self.fc1 = nn.Linear(model_dimension, model_dimension)

  def forward(self, x, **kwargs):
    x = self.fc1(x).relu()

    return {'x': x}




