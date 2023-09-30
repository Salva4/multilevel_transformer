import torch
import torch.nn as nn

from utils import get_batch

def train(_vars):
  _vars.optimizer.zero_grad()
  ctr_batch = 0
  _vars.model.train()

  while 1:
    batch = get_batch(_vars)
    output = _vars.model(batch[:, :-1])
    loss = _vars.criterion(output.transpose(1,2), batch[:, 1:])
    loss.backward()

    ctr_batch += 1
    if ctr_batch%10 == 0: 
      nn.utils.clip_grad_norm_(_vars.model.parameters(), .1)
      _vars.optimizer.step()
      _vars.optimizer.zero_grad()

    if ctr_batch%_vars.nmonitor == 0:
      _vars.model.eval()
      prompt = "L'emperadriu se aparella de anar per cercar lo bon comte de "
      with torch.no_grad():
        batch = torch.full((1, _vars.chunk_size), _vars.voc['<pad>'], 
                                      dtype=torch.long).to(_vars.dev)
        batch[0, :len(prompt)] = torch.LongTensor([_vars.voc[char] \
                                              for char in prompt])
        for step in range(len(prompt), _vars.chunk_size):
          output = _vars.model(batch)
          pred = (output.argmax(-1)[0, step])
          batch[0, step] = pred

        print(f'Training loss: {loss.item()}')
        print(''.join([_vars.voc[i] for i in batch[0].tolist()]))

      _vars.model.train()








































    
