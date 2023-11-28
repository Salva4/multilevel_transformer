import torch.nn as nn

from ..model_utils.positional_encoding import TorchPositionalEncoding

class PreContinuousBlock(nn.Module):
  def __init__(
    self, vocabulary_size, model_dimension, pad_token_id, **kwargs,
  ):
    super().__init__()#**kwargs)
    self.model_dimension = model_dimension
    self.pad_token_id = pad_token_id

    self.embedding = nn.Embedding(vocabulary_size, model_dimension)
    # self.dropout = nn.Dropout(p=.1)
    self.positional_encoding = TorchPositionalEncoding(self.model_dimension)

  def forward(self, x, **kwargs):
    padding_mask = (x == self.pad_token_id)#0)
    x = self.embedding(x)
    # x = self.dropout(x)
    x = self.positional_encoding(x)

    return {'x': x, 'padding_mask': padding_mask}




