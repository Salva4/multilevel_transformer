## Taken from Karpathy's github: [url]

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
  def __init__(self, head_size, d_model, block_size, dropout):
    super().__init__()
    self.key   = nn.Linear(d_model, head_size, bias=False)
    self.query = nn.Linear(d_model, head_size, bias=False)
    self.value = nn.Linear(d_model, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, 
                                                     block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    B,T,C = x.shape
    k = self.key(x)   # (B,T,hs)
    q = self.query(x) # (B,T,hs)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,hs)
    out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
    return out

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

class FeedForward(nn.Module):
  def __init__(self, d_model, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, 4 * d_model),
      nn.ReLU(),
      nn.Linear(4 * d_model, d_model),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, d_model, n_head, block_size, dropout):
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

class DTransformer(nn.Module):
  def __init__(self, d_model, n_head, num_layers, dropout, vocab_size, 
                              block_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, d_model)
    self.posenc = nn.Embedding(block_size, d_model)
    self.blocks = nn.Sequential(*[Block(d_model, n_head, block_size, 
                   dropout) for _ in range(num_layers)])
    # block = Block(d_model, n_head, block_size, dropout)
    # self.blocks = nn.Sequential(*[copy.deepcopy(block) \
    #                       for _ in range(num_layers)])
    self.ln = nn.LayerNorm(d_model) # final layer norm
    self.classifier = nn.Linear(d_model, vocab_size)

    # better init, not covered in the original GPT video, but important, will cover in followup video
    self.apply(self._init_weights)

    self.block_size = block_size

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, x, targets=None):
    B, T = x.shape
    positions = torch.arange(T, device=x.device)
    x = self.emb(x) + self.posenc(positions)  # (B,T,C) + (T,C) = (B,T,C)
    x = self.blocks(x) # (B,T,C)
    x = self.ln(x) # (B,T,C)
    logits = self.classifier(x) # (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, x, max_new_tokens):
    for _ in range(max_new_tokens):
      x_cond = x[:, -self.block_size:]
      logits, loss = self(x_cond)
      logits = logits[:, -1, :] # becomes (B, C)
      probs = F.softmax(logits, dim=-1) # (B, C)
      x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      x = torch.cat((x, x_next), dim=1) # (B, T+1)
    return x



















