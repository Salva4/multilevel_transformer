import torch.nn as nn

class PostContinuousBlock(nn.Module):
  def __init__(self, model_dimension, num_classes, **kwargs):
    super().__init__()
    self.fc2 = nn.Linear(model_dimension, num_classes)

  def forward(self, x, **kwargs):
    x = self.fc2(x)

    return {'x': x}




