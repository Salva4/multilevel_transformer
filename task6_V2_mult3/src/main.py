import numpy as np
import time
import torch 
import torch.nn as nn

from posenc import PositionalEncoding

class Voc:
  def __init__(self):
    self.specials = '<pad> <unk> <sos> <eos>'.split()
    self.str2id = dict(zip(self.specials, range(len(self.specials))))
    self.id2str = {v:k for (k,v) in self.str2id.items()}
  def __len__(self): return len(self.str2id)
  def __getitem__(self, x): return self.str2id[x] if type(x)==str else self.id2str[x]
  def add(self, x): 
    self.str2id[x] = len(self.str2id)
    self.id2str[len(self.id2str)] = x

class Transformer(nn.Module):
  def __init__(self, voc, d):
    super(Transformer, self).__init__()
    self.emb_src = nn.Embedding(len(voc['src']), d)
    self.emb_tgt = nn.Embedding(len(voc['tgt']), d)
    self.posenc = PositionalEncoding(d)
    self.transformer = nn.Transformer(d)
    self.fc = nn.Linear(d, len(voc['tgt']))
    self.voc = voc
  def forward(self, src, tgt):
    n, L = tgt.shape
    mask_src_pad = (src==voc['src']['<pad>'])
    mask_tgt_pad = (tgt==voc['tgt']['<pad>'])
    mask_tgt_att = nn.Transformer.generate_square_subsequent_mask(L).to(src.device)
    src, tgt = src.T, tgt.T
    src, tgt = self.emb_src(src), self.emb_tgt(tgt)
    src, tgt = self.posenc(src), self.posenc(tgt)
    out = self.transformer(src, tgt, tgt_mask=mask_tgt_att, src_key_padding_mask=mask_src_pad, 
      tgt_key_padding_mask=mask_tgt_pad, memory_key_padding_mask=mask_src_pad)
    out = self.fc(out)
    return out.transpose(0,1)

def corr_tot_(pred, tgt_out, voc):
  corr = ((pred==tgt_out) + (tgt_out==voc['tgt']['<pad>'])).prod(-1).int().sum()
  tot = pred.shape[0]
  return np.array([corr, tot])

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

t0 = time.time(); print('Generating dataset... ', end='')
## Obtain data
ds_raw = []
N = 10000
for x in range(N):
  q, a = str(x), 'yes' if x%3==0 else 'no'
  ds_raw.append((q, a))

voc = {}; voc['src'] = Voc(); voc['tgt'] = Voc()
ds = []
for (q_raw, a_raw) in ds_raw: # ('6', 'yes')
  q, a = [], []
  for char in q_raw:
    if char not in voc['src'].str2id: voc['src'].add(char)
    q.append(voc['src'][char])
  a.append(voc['tgt']['<sos>'])
  for char in a_raw:
    if char not in voc['tgt'].str2id: voc['tgt'].add(char)
    a.append(voc['tgt'][char])
  a.append(voc['tgt']['<eos>'])
  for _ in range((len(str(N))-1)-len(q)): q.append(voc['src']['<pad>'])
  for _ in range(5-len(a)): a.append(voc['tgt']['<pad>'])
  q, a = torch.LongTensor(q).to(dev), torch.LongTensor(a).to(dev)
  ds.append((q,a))

np.random.shuffle(ds)
ds = {'tr': ds[:int(.8*N)], 'va': ds[int(.8*N):]}
print(f'{time.time() - t0 :.2f}s')

## Setup
dl = {part: torch.utils.data.DataLoader(_ds, batch_size=2, shuffle=True, drop_last=True) \
                                                           for (part, _ds) in ds.items()}
model = Transformer(voc, 128).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=voc['tgt']['<pad>'])

## Training
for _ in range(1000000000):#1000):
  ## Training partition
  model.train()
  losses = []
  corr_tot = np.array([0, 0])
  for (src, tgt) in dl['tr']:
    tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:].cpu()
    out = model(src, tgt_inp).cpu()
    loss = criterion(out.transpose(1, 2), tgt_out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      losses.append(loss.item())
      pred = out.argmax(-1)
      # corr += ((pred==tgt_out)*(tgt_out!=voc['tgt']['<pad>'])).sum()
      # tot += (tgt_out!=voc['tgt']['<pad>']).sum()
      corr_tot += corr_tot_(pred, tgt_out, voc)

  loss_tr = np.mean(losses)
  acc_tr = corr_tot[0]/corr_tot[1]
  print(f'Loss_tr: {loss_tr:.4f}, Acc_tr: {100*acc_tr:.2f}%', end=', ')

  ## Validation partition
  losses = []
  corr_tot = np.array([0, 0])
  with torch.no_grad():
    for (src, tgt) in dl['va']:
      tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:].cpu()
      out = model(src, tgt_inp).cpu()
      loss = criterion(out.transpose(1, 2), tgt_out)
      losses.append(loss.item())
      pred = out.argmax(-1)
      corr_tot += corr_tot_(pred, tgt_out, voc)

  loss_va = np.mean(losses)
  acc_va = corr_tot[0]/corr_tot[1]
  print(f'Loss_va: {loss_va:.4f}, Acc_va: {100*acc_va:.2f}%')

  ## Update lr
  # for g in optimizer.param_groups:
  #   g['lr'] /= 1.5























