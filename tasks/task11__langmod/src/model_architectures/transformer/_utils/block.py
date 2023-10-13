import torch.nn as nn

from .feed_forward_network import FeedForward
from .multi_head_attention import MultiHeadAttention

class Block(nn.Module):
  def __init__(self, d_model, n_head, block_size, dropout, **kwargs):
    # d_model: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = d_model // n_head
    self.sa = MultiHeadAttention(n_head, head_size, d_model, block_size, 
                                 dropout)
    self.ffwd = FeedForward(d_model, dropout)
    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
