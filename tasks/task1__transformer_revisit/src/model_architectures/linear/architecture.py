import torch
import torch.nn as nn

# ##
# # Trivial 2-layer linear classifier
# class Linear(nn.Module):    
#   def __init__(self, **kwargs):
#     super(Linear, self).__init__()
#     self.emb = nn.Embedding(15514, 256)
#     self.fc1 = nn.Linear(256, 64)
#     self.fc2 = nn.Linear(64, 49)

#   def forward(self, x):
#     x = self.emb(x)
#     x = self.fc1(x).relu()
#     x = self.fc2(x)
#     return x

class ContinuousResidualLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(256, 256)

  def forward(self, x, **kwargs):
    x = self.fc1(x).relu()

    return {'x': x}

class PostContinuousBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc2 = nn.Linear(256, 49)

  def forward(self, x, **kwargs):
    x = self.fc2(x)

    return {'x': x}

class PreContinuousBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(15514, 256)

  def forward(self, x, **kwargs): 
    x = self.emb(x)

    return {'x': x}



