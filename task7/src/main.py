## Example taken from https://www.youtube.com/watch?v=M6adRGJe5cQ&t=93s

## Requires: torchtext==0.8, spacy w/ english and deutsch

import spacy
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from torchtext.datasets import Multi30k

langs, parts = ['de', 'en'], ['tr', 'va', 'te']
tokenizer = {'de': spacy.load('de_core_news_sm').tokenizer, 
             'en': spacy.load('en_core_web_sm').tokenizer}
field = {lang: Field(tokenize=lambda s: [token.text \
                     for token in tokenizer[lang](s)], lower=True, 
                     init_token='<sos>', eos_token='<eos>') for lang in langs}
voc_inv = {lang: {v:k for (k,v) in field[lang].vocab.stoi.items()} 
                                                for lang in langs}
data = dict(zip(parts, Multi30k.splits(exts=('.de','.en'), fields=[field[lang] \
                                                         for lang in langs])))

for lang in langs: field[lang].build_vocab(data['tr'], max_size=10000, 
                                                           min_freq=2)

class Transformer(nn.Module):
  def __init__(self, d, dim_voc_src, dim_voc_tgt, idx_pad_src, nheads,
               n_lays_enc, n_lays_dec, dropout, max_len, dev):
    super(Transformer, self).__init__()
    self.emb_src = nn.Embedding(dim_voc_src, d)
    self.emb_tgt = nn.Embedding(dim_voc_tgt, d)
    self.posenc_src = nn.Embedding(max_len, d)
    self.posenc_tgt = nn.Embedding(max_len, d)
    self.dev = dev
    self.transformer = nn.Transformer(d, nheads, n_lays_enc, n_lays_dec)
    self.classifier = nn.Linear(d, dim_voc_tgt)
    self.dropout = nn.Dropout(dropout)
    self.idx_pad_src = idx_pad_src

  def forward(self, src, tgt):
    L_src, n = src.shape
    L_tgt, _ = tgt.shape
    mask_pad_src = (src.T == self.idx_pad_src)
    # mask_pad_tgt = (tgt.T == self.idx_pad_tgt)
    mask_att_tgt = nn.Transformer.generate_square_subsequent_mask(L_tgt).to(
                                                                   self.dev)
    # src, tgt = src.T, tgt.T
    pos_src, pos_tgt = [torch.arange(L).unsqueeze(1).expand(-1, n).to(
                                    self.dev) for L in [L_src, L_tgt]]
    src, tgt = (self.dropout(self.emb_src(src) + self.posenc_src(pos_src)),
                self.dropout(self.emb_tgt(tgt) + self.posenc_tgt(pos_tgt)))
    out = self.transformer(src, tgt, src_key_padding_mask=mask_pad_src, tgt_mask=mask_att_tgt)
    out = self.classifier(out)
    return out#.transpose(0,1)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print(dev)
n_epochs = 5
lr = 3e-4
batch_size = 32
dim_voc_src, dim_voc_tgt = [len(field[lang].vocab) for lang in langs]
d = 512
nheads = 8
n_lays_enc, n_lays_dec = 3, 3
dropout = .1
max_len = 100
idx_pad_src, idx_pad_tgt = [field[lang].vocab.stoi['<pad>'] for lang in langs]

## Tensorboard
# writer = SummaryWriter('runs/loss_plot')
step = 0

it_tr, it_va, it_te = BucketIterator.splits([data[part] for part in parts],
                             batch_size=batch_size, sort_within_batch=True, 
                                 sort_key=lambda x: len(x.src), device=dev)
model = Transformer(d, dim_voc_src, dim_voc_tgt, idx_pad_src, nheads,
         n_lays_enc, n_lays_dec, dropout, max_len, dev).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=idx_pad_tgt)
sentence = 'ich möchte diesen film sehen.'

for epoch in range(n_epochs):
  model.train()
  losses = []
  for batch in it_tr:
    src, tgt = batch.src.to(dev), batch.trg.to(dev)
    out = model(src, tgt[:-1])
    loss = criterion(out.transpose(1,2), tgt[1:])
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    losses.append(loss.item())
  print(f'Epoch {epoch}/{n_epochs}, Loss_tr: {np.mean(losses):.4f}, ', end='')

  model.eval()
  with torch.no_grad():
    losses, candidate_corpus, reference_corpus = [], [], []
    for batch in it_va:
      src, tgt = batch.src.to(dev), batch.trg.to(dev)
      out = model(src, tgt[:-1])
      loss = criterion(out.transpose(1,2), tgt[1:])
      losses.append(loss.item())

      preds = out.argmax(dim=-1)
      for pred, tgt_ in zip(preds.T, (tgt[1:]).T):
        pred, tgt_ = (' '.join([voc_inv['en'][i.item()] for i in tensor]) for tensor in (pred, tgt_))
        candidate_corpus.append(pred.split())
        reference_corpus.append([tgt_.split()])

    bleu = bleu_score(candidate_corpus, reference_corpus)

  print(f'Loss_va: {np.mean(losses):.4f}, Bleu_va: {bleu}')
  






















































