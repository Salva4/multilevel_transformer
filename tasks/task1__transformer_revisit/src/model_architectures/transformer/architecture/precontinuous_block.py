import torch.nn as nn

from ..model_utils.positional_encoding import TorchPositionalEncoding

class PreContinuousBlock(nn.Module):
  def __init__(
    self, vocabulary_size, model_dimension, pad_token_id, **kwargs,
  ):
    super().__init__()#**kwargs)
    self.Σ_size = vocabulary_size
    self.d = model_dimension
    self.pad_token_id = pad_token_id

    self.emb = nn.Embedding(vocabulary_size, model_dimension)#15514, self.d)
    # self.dropout = nn.Dropout(p=.1)
    self.posenc = TorchPositionalEncoding(self.d)

  def forward(self, x, **kwargs):
    padding_mask = (x == self.pad_token_id)#0)
    x = self.emb(x)
    # x = self.dropout(x)
    x = self.posenc(x)

    return {'x': x, 'padding_mask': padding_mask}




