import torch.nn as nn

from ._utils.positional_encoding import TorchPositionalEncoding

class ContinuousResidualLayer(nn.Module):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)    
    self.fc1 = nn.Linear(128, 128)
    self.fc2 = nn.Linear(128, 128)
    self.att = nn.MultiheadAttention(
      embed_dim=128, 
      num_heads=1, 
      dropout=0,#.3, 
      batch_first=True
    )
    self.ln1 = nn.LayerNorm(128)
    self.ln2 = nn.LayerNorm(128)

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
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.fc3 = nn.Linear(128, 49)
    self.ln3 = nn.LayerNorm(128)

  def forward(self, x, **kwargs):
    x = self.ln3(x)
    x = self.fc3(x)

    return {'x': x}

class PreContinuousBlock(nn.Module):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.emb = nn.Embedding(15514, 128)
    # self.dropout = nn.Dropout(p=.1)
    self.posenc = TorchPositionalEncoding(128)

  def forward(self, x, **kwargs):
    padding_mask = (x == 0)
    x = self.emb(x)
    # x = self.dropout(x)
    x = self.posenc(x)

    return {'x': x, 'padding_mask': padding_mask}



