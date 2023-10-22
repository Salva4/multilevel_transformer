import torch.nn as nn

from ._utils.positional_encoding import TorchPositionalEncoding

class PreContinuousBlock(nn.Module):
  def __init__(self, vocabulary_size, model_dimension, **kwargs):
    super().__init__()#**kwargs)
    self.Σ_size = vocabulary_size
    self.d = model_dimension

    self.emb = nn.Embedding(15514, self.d)
    # self.dropout = nn.Dropout(p=.1)
    self.posenc = TorchPositionalEncoding(self.d)

  def forward(self, x, **kwargs):
    padding_mask = (x == 0)
    x = self.emb(x)
    # x = self.dropout(x)
    x = self.posenc(x)

    return {'x': x, 'padding_mask': padding_mask}
    
class ContinuousResidualLayer(nn.Module):
  def __init__(self, model_dimension, num_heads, **kwargs):
    '''
    Default: model_dimension=128, num_heads=1, dropout=.3 (changed to 0. in tb)
    '''
    super().__init__()#**kwargs)
    self.d = model_dimension
    self.num_heads = num_heads

    self.fc1 = nn.Linear(self.d, self.d)
    self.fc2 = nn.Linear(self.d, self.d)
    self.att = nn.MultiheadAttention(
      embed_dim=self.d, 
      num_heads=self.num_heads, 
      dropout=0,#.3, 
      batch_first=True
    )
    self.ln1 = nn.LayerNorm(self.d)
    self.ln2 = nn.LayerNorm(self.d)

  def forward(self, x, padding_mask, **kwargs):
    # Encoder1DBlock
    x0 = x
    x = self.ln1(x)
    x, _ = self.att(x, x, x, padding_mask)
    # x = self.dropout(x)
    x1 = x
    x = x + x0
    
    x = self.ln2(x)
    # MLPBlock
    x = self.fc1(x)
    x = nn.ELU()(x)
    x = self.fc2(x)

    x = x + x1

    return {'x': x}

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


