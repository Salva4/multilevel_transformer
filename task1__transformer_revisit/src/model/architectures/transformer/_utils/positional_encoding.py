import math
import numpy as np
import torch
import torch.nn as nn

##
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class TorchPositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.0, max_len=5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.max_len = max_len

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
               * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
    self.register_buffer('pe', pe)  # Will not be trained.

  def forward(self, x):
    """Inputs of forward function
    Args:
      x: the sequence fed to the positional encoder model (required).
    Shape:
      x: [sequence length, batch size, embed dim]
      output: [sequence length, batch size, embed dim]
    """
    assert x.size(0) < self.max_len, (
      f"Too long sequence length: increase `max_len` of pos encoding")
    # shape of x (len, B, dim)
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

# ##
# # Taken from: 
# # https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
# class PE_Alternative(nn.Module):
#   def __init__(self, d_model, dropout=0.0, max_len=5000):
#     super(PE_Alternative, self).__init__()
#     self.max_len = max_len

#     pe = torch.zeros(max_len, d_model)
#     for k in range(max_len):
#       for i in np.arange(d_model//2):
#         denominator = np.power(10000, 2*i/d_model)
#         pe[k, 2*i] = np.sin(k/denominator)
#         pe[k, 2*i + 1] = np.cos(k/denominator)

#     # self.register_buffer('pe', pe)  # Will not be trained.
#     self.pe = pe

#   def forward(self, x):
#     assert x.shape[1] < self.max_len, (
#       f"Too long sequence length: increase `max_len` of pos encoding")
#     # shape of x (len, B, dim)
#     x = x + self.pe[:x.shape[1], :].unsqueeze(dim=0)
#     return x
