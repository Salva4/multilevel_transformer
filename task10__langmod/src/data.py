import torch

path_data = '/users/msalvado/prova_transfpytorch/task10__langmod/data/cronica_Bernat_Desclot.txt'
# path_data = '/users/msalvado/prova_transfpytorch/task10__langmod/data/tiny_shakespeare.txt'

class Voc:
  def __init__(self):
    self.specials = '<pad> <unk> <sos> <eos>'.split()
    self.str2id = dict(zip(self.specials, range(len(self.specials))))
    self.id2str = {v:k for (k,v) in self.str2id.items()}

  def __getitem__(self, x): return self.str2id.get(x, self.str2id['<unk>']) \
                                        if type(x)==str else self.id2str[x]

  def __len__(self): return len(self.str2id)

  def add(self, x):
    self.str2id[x] = len(self.str2id)
    self.id2str[len(self.id2str)] = x

def obtain_data(_vars):
  with open(path_data, 'r') as f:
    t = f.read()

  data = []
  voc = Voc()

  for char in t: 
    if char not in voc.str2id: voc.add(char)
    data.append(voc[char])

  _vars.data = torch.LongTensor(data)
  _vars.voc = voc







































