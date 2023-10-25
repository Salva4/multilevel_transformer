import numpy as np
import torch
import torch.nn as nn

from .continuous_block import ContinuousBlock

##
# Continuous transformer encoder layer using their code's scheme & <i>MultiheadAttention</i>
class ContinuousModel(nn.Module):
  def __init__(self, model, **kwargs_continuous_block):#, num_levels):
    print('Continuous approach')
    super().__init__()
    self.model = model
    # self.register_buffer('model', model)
    # self.interpol = kwargs['interpol']
    print(f'''N={self.model.continuous_block.N}, T={kwargs_continuous_block['T']}''')
    print()

    self.precontinuous_block  = self.model.precontinuous_block
    self.postcontinuous_block = self.model.postcontinuous_block

    self.continuous_block = ContinuousBlock(
      # ψ=self.model.continuous_block.residual_layers,
      ψ=nn.ModuleList(
        [layer.residual_layer for layer in model.continuous_block.layers]
      ),
      Nf=self.model.continuous_block.N,
      **kwargs_continuous_block,
      # num_levels=num_levels,
    )
      # interpol=self.interpol,

    # if self.init_method != 'None':
    #   self.init_params()

    # self.num_params = self.weights_flat_().shape[0]

  def forward(self, **state):
    # state = {'x': x}
    # state.update(self.precontinuous_block (**state))
    # state.update(self.continuous_block    (**state))
    # state.update(self.postcontinuous_block(**state))
    # return state
    return self.model.static_forward(self, **state)

  def save(self, **kwargs): 
    '''Arguments: fn_without_extension=None, models_dir=None, optimizer=None, 
                  **other''' 
    self.model.static_save(self, **kwargs)

  # def init_params(self):
  #   self.apply(self._init_layers)

  # def _init_layers(self, m):
  #   classname = m.__class__.__name__
  #   if isinstance(m, nn.Conv2d):
  #     if m.weight is not None:
  #       torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

  #     if m.bias is not None:
  #       torch.nn.init.constant_(m.bias, 0)
    
  #   if isinstance(m, nn.BatchNorm2d):
  #     if m.weight is not None:
  #       torch.nn.init.constant_(m.weight, 1)

  #     if m.bias is not None:
  #       torch.nn.init.constant_(m.bias, 0)
    
  #   if isinstance(m, nn.Linear):
  #     if m.weight is not None:
  #       torch.nn.init.normal_(m.weight)

  #     if m.bias is not None:
  #       torch.nn.init.constant_(m.bias, 0)
    
  #   if isinstance(m, nn.MultiheadAttention):
  #     if self.init_method == 'Normal':
  #       m.in_proj_weight.data.normal_(mean=0.0, std=0.02)
  #       m.out_proj.weight.data.normal_(mean=0.0, std=0.02)
  #     elif self.init_method == 'Xavier':
  #       torch.nn.init.xavier_uniform_(m.in_proj_weight, gain=np.sqrt(2))
  #       torch.nn.init.xavier_uniform_(m.out_proj.weight, gain=np.sqrt(2))
  #     else:
  #       Exception('Initialization method unknown')

  # # def init_weights_from_model(self, old_model):     <-- not used in MG/OPT
  # #   self.emb.weight.data = old_model.emb.weight.data
  # #   self.continuous_block.init_weights_from_model(old_model)
  # #   self.ln3.weight.data = old_model.ln3.weight.data
  # #   self.ln3.bias.data = old_model.ln3.bias.data
  # #   self.fc3.weight.data = old_model.fc3.weight.data
  # #   self.fc3.bias.data = old_model.fc3.bias.data



  # def interpolate_weights_from(self, old_model, lr):
  #   ## Pre
  #   self.emb.weight.data += lr*old_model.emb.weight.data

  #   ## Cont
  #   self.continuous_block.interpolate_weights_from(old_model, lr)

  #   ## Post
  #   self.ln3.weight.data += lr*old_model.ln3.weight.data
  #   self.ln3.bias.data += lr*old_model.ln3.bias.data
  #   self.fc3.weight.data += lr*old_model.fc3.weight.data
  #   self.fc3.bias.data += lr*old_model.fc3.bias.data

  # def restrict_weights_from(self, old_model):
  #   ## Pre
  #   self.emb.weight.data = old_model.emb.weight.data.clone()

  #   ## Cont
  #   self.continuous_block.restrict_weights_from(old_model)

  #   ## Post
  #   self.ln3.weight.data = old_model.ln3.weight.data.clone()
  #   self.ln3.bias.data = old_model.ln3.bias.data.clone()
  #   self.fc3.weight.data = old_model.fc3.weight.data.clone()
  #   self.fc3.bias.data = old_model.fc3.bias.data.clone()

  # def update_diff_weights(self, old_model):
  #   ## Pre
  #   self.emb.weight.data -= old_model.emb.weight.data

  #   ## Cont
  #   self.continuous_block.update_diff_weights(old_model)

  #   ## Post
  #   self.ln3.weight.data -= old_model.ln3.weight.data
  #   self.ln3.bias.data -= old_model.ln3.bias.data
  #   self.fc3.weight.data -= old_model.fc3.weight.data
  #   self.fc3.bias.data -= old_model.fc3.bias.data

  # def weights_flat_(self):
  #   weights = torch.cat((self.emb.weight.data.flatten(),
  #              torch.cat([torch.cat([self.continuous_block.Rs[n].fc1.weight.data.flatten(),
  #                    self.continuous_block.Rs[n].fc1.bias.data.flatten(),
  #                    self.continuous_block.Rs[n].fc2.weight.data.flatten(),
  #                    self.continuous_block.Rs[n].fc2.bias.data.flatten(),
  #                    self.continuous_block.Rs[n].att.in_proj_weight.data.flatten(),
  #                    self.continuous_block.Rs[n].att.in_proj_bias.data.flatten(),
  #                    self.continuous_block.Rs[n].att.out_proj.weight.data.flatten(),
  #                    self.continuous_block.Rs[n].att.out_proj.bias.data.flatten(),
  #                    self.continuous_block.Rs[n].ln1.weight.data.flatten(),
  #                    self.continuous_block.Rs[n].ln1.bias.data.flatten(),
  #                    self.continuous_block.Rs[n].ln2.weight.data.flatten(),
  #                    self.continuous_block.Rs[n].ln2.bias.data.flatten()], axis=0)
  #                    for n in range(self.N)], axis=0),
  #              self.ln3.weight.data.flatten(),
  #              self.ln3.bias.data.flatten(),
  #              self.fc3.weight.data.flatten(),
  #              self.fc3.bias.data.flatten()), 
  #                     axis=0)
  #   return weights

  # def grad_(self):
  #   if self.emb.weight.grad == None:
  #     return None
      
  #   grad = torch.cat((self.emb.weight.grad.flatten(),
  #              torch.cat([torch.cat([self.continuous_block.Rs[n].fc1.weight.grad.flatten(),
  #                    self.continuous_block.Rs[n].fc1.bias.grad.flatten(),
  #                    self.continuous_block.Rs[n].fc2.weight.grad.flatten(),
  #                    self.continuous_block.Rs[n].fc2.bias.grad.flatten(),
  #                    self.continuous_block.Rs[n].att.in_proj_weight.grad.flatten(),
  #                    self.continuous_block.Rs[n].att.in_proj_bias.grad.flatten(),
  #                    self.continuous_block.Rs[n].att.out_proj.weight.grad.flatten(),
  #                    self.continuous_block.Rs[n].att.out_proj.bias.grad.flatten(),
  #                    self.continuous_block.Rs[n].ln1.weight.grad.flatten(),
  #                    self.continuous_block.Rs[n].ln1.bias.grad.flatten(),
  #                    self.continuous_block.Rs[n].ln2.weight.grad.flatten(),
  #                    self.continuous_block.Rs[n].ln2.bias.grad.flatten()], axis=0)
  #                    for n in range(self.N)], axis=0),
  #              self.ln3.weight.grad.flatten(),
  #              self.ln3.bias.grad.flatten(),
  #              self.fc3.weight.grad.flatten(),
  #              self.fc3.bias.grad.flatten()), 
  #                     axis=0)
  #   return grad

  # def Rxgrad_(self):
  #   # cont_grads = [[ self.continuous_block.Rs[n].fc1.weight.grad,
  #   #         self.continuous_block.Rs[n].fc1.bias.grad,
  #   #         self.continuous_block.Rs[n].fc2.weight.grad,
  #   #         self.continuous_block.Rs[n].fc2.bias.grad,
  #   #         self.continuous_block.Rs[n].att.in_proj_weight.grad,
  #   #         self.continuous_block.Rs[n].att.in_proj_bias.grad,
  #   #         self.continuous_block.Rs[n].att.out_proj.weight.grad,
  #   #         self.continuous_block.Rs[n].att.out_proj.bias.grad,
  #   #         self.continuous_block.Rs[n].ln1.weight.grad,
  #   #         self.continuous_block.Rs[n].ln1.bias.grad,
  #   #         self.continuous_block.Rs[n].ln2.weight.grad,
  #   #         self.continuous_block.Rs[n].ln2.bias.grad
  #   #         ] for n in range(self.N)]

  #   # Rxgrad = [
  #   #   self.emb.weight.grad,
  #   #   [
  #   #     [
  #   #       1/2*(
  #   #           (1/2*cont_grads[2*n - 1][iw] if n > 0 else 0) +\
  #   #           cont_grads[2*n][iw] +\
  #   #           1/2*cont_grads[2*n + 1][iw]
  #   #       ) for iw in range(len(cont_grads[0]))
  #   #     ] for n in range(self.N//2)
  #   #   ],
  #   #   self.ln3.weight.grad,
  #   #   self.ln3.bias.grad,
  #   #   self.fc3.weight.grad,
  #   #   self.fc3.bias.grad, 
  #   # ]
  #   cont_grads = [[self.continuous_block.Rs[n].fc1.weight.grad.flatten(),
  #           self.continuous_block.Rs[n].fc1.bias.grad.flatten(),
  #           self.continuous_block.Rs[n].fc2.weight.grad.flatten(),
  #           self.continuous_block.Rs[n].fc2.bias.grad.flatten(),
  #           self.continuous_block.Rs[n].att.in_proj_weight.grad.flatten(),
  #           self.continuous_block.Rs[n].att.in_proj_bias.grad.flatten(),
  #           self.continuous_block.Rs[n].att.out_proj.weight.grad.flatten(),
  #           self.continuous_block.Rs[n].att.out_proj.bias.grad.flatten(),
  #           self.continuous_block.Rs[n].ln1.weight.grad.flatten(),
  #           self.continuous_block.Rs[n].ln1.bias.grad.flatten(),
  #           self.continuous_block.Rs[n].ln2.weight.grad.flatten(),
  #           self.continuous_block.Rs[n].ln2.bias.grad.flatten()]
  #           for n in range(self.N)]

  #   #### No injection:
  #   # Rxgrad = torch.cat((self.emb.weight.grad.flatten(),
  #   #           torch.cat([torch.cat([1/2*(1/2*cont_grads[2*n - 1][iw] if n > 0 else 0) +\
  #   #                       cont_grads[2*n][iw] +\
  #   #                       1/2*cont_grads[2*n + 1][iw]
  #   #                       for iw in range(len(cont_grads[0]))], axis=0)
  #   #                  for n in range(self.N//2)], axis=0),
  #   #           self.ln3.weight.grad.flatten(),
  #   #           self.ln3.bias.grad.flatten(),
  #   #           self.fc3.weight.grad.flatten(),
  #   #           self.fc3.bias.grad.flatten()), 
  #   #               axis=0)
  #   ## Yes injection:
  #   Rxgrad = torch.cat((self.emb.weight.grad.flatten(),
  #             torch.cat([torch.cat([cont_grads[2*n][iw]
  #                         for iw in range(len(cont_grads[0]))], axis=0)
  #                    for n in range(self.N//2)], axis=0),
  #             self.ln3.weight.grad.flatten(),
  #             self.ln3.bias.grad.flatten(),
  #             self.fc3.weight.grad.flatten(),
  #             self.fc3.bias.grad.flatten()), 
  #                 axis=0)
  #   #### 
  #   return Rxgrad

  # def add2wgrad(self, dg):
  #   # idx = 0
    
  #   # self.emb.weight.grad.data += dg[idx : idx + 15514*128].reshape(15514, 128)
  #   # idx += 15514*128

  #   # for n in range(self.continuous_block.N):
  #   #   self.continuous_block.Rs[n].fc1.weight.grad.data += dg[idx : idx + 128*128].reshape(128, 128)
  #   #   idx += 128*128
  #   #   self.continuous_block.Rs[n].fc1.bias.grad.data += dg[idx : idx + 128]
  #   #   idx += 128
  #   #   self.continuous_block.Rs[n].fc2.weight.grad.data += dg[idx : idx + 128*128].reshape(128, 128)
  #   #   idx += 128*128
  #   #   self.continuous_block.Rs[n].fc2.bias.grad.data += dg[idx : idx + 128]
  #   #   idx += 128
  #   #   self.continuous_block.Rs[n].att.in_proj_weight.grad.data += dg[idx: idx + 384*128].reshape(384, 128)
  #   #   idx += 384*128
  #   #   self.continuous_block.Rs[n].att.in_proj_bias.grad.data += dg[idx: idx + 384]
  #   #   idx += 384
  #   #   self.continuous_block.Rs[n].att.out_proj.weight.grad.data += dg[idx: idx + 128*128].reshape(128, 128)
  #   #   idx += 128*128
  #   #   self.continuous_block.Rs[n].att.out_proj.bias.grad.data += dg[idx: idx + 128]
  #   #   idx += 128
  #   #   self.continuous_block.Rs[n].ln1.weight.grad.data += dg[idx: idx + 128]
  #   #   idx += 128
  #   #   self.continuous_block.Rs[n].ln1.bias.grad.data += dg[idx: idx + 128]
  #   #   idx += 128
  #   #   self.continuous_block.Rs[n].ln2.weight.grad.data += dg[idx: idx + 128]
  #   #   idx += 128
  #   #   self.continuous_block.Rs[n].ln2.bias.grad.data += dg[idx: idx + 128]
  #   #   idx += 128

  #   # self.ln3.weight.grad.data += dg[idx: idx + 128]
  #   # idx += 128
  #   # self.ln3.bias.grad.data += dg[idx: idx + 128]
  #   # idx += 128
  #   # self.fc3.weight.grad.data += dg[idx: idx + 49*128].reshape(49, 128)
  #   # idx += 49*128
  #   # self.fc3.bias.grad.data += dg[idx: idx + 49]
  #   # idx += 49
  #   # assert idx == dg.shape[0] and dg.ndim == 1

  #   ## Sad:
  #   if type(dg) == list:
  #     dg = iter(dg)

  #   self.emb.weight.grad.data += next(dg)

  #   for n in range(self.continuous_block.N):
  #     self.continuous_block.Rs[n].fc1.weight.grad.data += next(dg)
  #     self.continuous_block.Rs[n].fc1.bias.grad.data += next(dg)
  #     self.continuous_block.Rs[n].fc2.weight.grad.data += next(dg)
  #     self.continuous_block.Rs[n].fc2.bias.grad.data += next(dg)
  #     self.continuous_block.Rs[n].att.in_proj_weight.grad.data += next(dg)
  #     self.continuous_block.Rs[n].att.in_proj_bias.grad.data += next(dg)
  #     self.continuous_block.Rs[n].att.out_proj.weight.grad.data += next(dg)
  #     self.continuous_block.Rs[n].att.out_proj.bias.grad.data += next(dg)
  #     self.continuous_block.Rs[n].ln1.weight.grad.data += next(dg)
  #     self.continuous_block.Rs[n].ln1.bias.grad.data += next(dg)
  #     self.continuous_block.Rs[n].ln2.weight.grad.data += next(dg)
  #     self.continuous_block.Rs[n].ln2.bias.grad.data += next(dg)

  #   self.ln3.weight.grad.data += next(dg)
  #   self.ln3.bias.grad.data += next(dg)
  #   self.fc3.weight.grad.data += next(dg)
  #   self.fc3.bias.grad.data += next(dg)

  #   assert next(dg, None) == None
