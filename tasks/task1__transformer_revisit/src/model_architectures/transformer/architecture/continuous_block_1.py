import torch.nn as nn

class ContinuousResidualLayer(nn.Module):
  def __init__(self, model_dimension, num_heads, **kwargs):
    '''
    Default: model_dimension=128, num_heads=1, dropout=.3 (changed to 0. so that mgrit can be applied)
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




