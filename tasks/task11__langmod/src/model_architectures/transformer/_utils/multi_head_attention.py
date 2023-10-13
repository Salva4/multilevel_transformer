import torch
import torch.nn as nn

from .head import Head

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size, d_model, block_size, dropout):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, d_model, block_size, 
                   dropout) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out