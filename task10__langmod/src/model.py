import torch
import torch.nn as nn

from utils import pass_args

class DTransformer(nn.Module):
  def __init__(self, _vars):
    super(DTransformer, self).__init__()
    self._vars = _vars
    self.emb = nn.Embedding(len(_vars.voc), _vars.d_model)
    self.posenc = nn.Embedding(_vars.chunk_size, _vars.d_model)
    _vars.encoder_layer = pass_args(nn.TransformerEncoderLayer, _vars)
    self.encoder = pass_args(nn.TransformerEncoder, _vars)
    self.classifier = nn.Linear(_vars.d_model, len(_vars.voc))
    self.to(_vars.dev)

  def forward(self, x):
    mask_pad = (x == self._vars.voc['<pad>'])
    mask_att = nn.Transformer.generate_square_subsequent_mask(
                     self._vars.chunk_size).to(self._vars.dev)
    pos = torch.arange(self._vars.chunk_size).repeat(x.shape[0], 1).to(
                                                            self._vars.dev)
    x = self.emb(x.T) + self.posenc(pos.T)
    x = self.encoder(src=x, mask=mask_att, src_key_padding_mask=mask_pad)
    x = self.classifier(x)
    return x.transpose(0,1)