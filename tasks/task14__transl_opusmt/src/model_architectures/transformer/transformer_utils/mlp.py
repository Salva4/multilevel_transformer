import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, model_dimension, dim_ff):
    super().__init__()
    self.fc1 = nn.Linear(model_dimension, dim_ff)
    self.fc2 = nn.Linear(dim_ff, model_dimension)
    self.activation_fn = nn.SiLU()

  def forward(self, x):
    _h = self.fc1(x)
    h = self.activation_fn(_h)
    o = self.fc2(h)
    return o
