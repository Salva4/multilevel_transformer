import torch
import torch.nn as nn
import copy

class ContinuousBlock(nn.Module):
  def __init__(self, φ, N, T, solver='Forward Euler'):#, interpol):
    super().__init__()
    self.φ = φ  # basis functions
    self.N = N
    self.T = T
    self.solver = solver
    # self.interpol = interpol

    self.dt = T/N

    # self.weights = [
    #   'fc1.weight', 'fc1.bias', 
    #   'fc2.weight', 'fc2.bias', 
    #   'att.in_proj_weight', 'att.in_proj_bias', 'att.out_proj.weight', 'att.out_proj.bias', 
    #   'ln1.weight', 'ln1.bias', 
    #   'ln2.weight', 'ln2.bias'
    # ]

  def forward(self, x, **kwargs):
    if self.solver == 'Forward Euler':
      for i in range(self.N):
        x = x + self.dt * self.φ[i](x, **kwargs)['x']

    else: raise Exception()

    return {'x': x}

  # def init_weights_from_model(self, old_model):
  #   self.interpolate_weights_from(old_model)

  # def interpolate_weights_from(self, old_model, lr):  # here old_model is coarser
  #   gen_params = old_model.parameters()

  #   for _ in range(1):
  #     _ = next(gen_params)

  #   ## Constant. Later change to linear
  #   for n_coarse in range(old_model.N):
  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].fc1.weight.data += lr * weights
  #     self.Rs[2*n_coarse + 1].fc1.weight.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].fc1.bias.data += lr * weights
  #     self.Rs[2*n_coarse + 1].fc1.bias.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].fc2.weight.data += lr * weights
  #     self.Rs[2*n_coarse + 1].fc2.weight.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].fc2.bias.data += lr * weights
  #     self.Rs[2*n_coarse + 1].fc2.bias.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].att.in_proj_weight.data += lr * weights
  #     self.Rs[2*n_coarse + 1].att.in_proj_weight.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].att.in_proj_bias.data += lr * weights
  #     self.Rs[2*n_coarse + 1].att.in_proj_bias.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].att.out_proj.weight.data += lr * weights
  #     self.Rs[2*n_coarse + 1].att.out_proj.weight.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].att.out_proj.bias.data += lr * weights
  #     self.Rs[2*n_coarse + 1].att.out_proj.bias.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].ln1.weight.data += lr * weights
  #     self.Rs[2*n_coarse + 1].ln1.weight.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].ln1.bias.data += lr * weights
  #     self.Rs[2*n_coarse + 1].ln1.bias.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].ln2.weight.data += lr * weights
  #     self.Rs[2*n_coarse + 1].ln2.weight.data += lr * weights

  #     weights = next(gen_params).data
  #     self.Rs[2*n_coarse].ln2.bias.data += lr * weights
  #     self.Rs[2*n_coarse + 1].ln2.bias.data += lr * weights

  #   for _ in range(4):
  #     _ = next(gen_params)

  #   assert next(gen_params, None) == None

  #   # for n_old in range(old_model.N):  # we undermine the last function weights
  #   #   for weight in self.weights:
  #   #     ## t_H --> t_h
  #   #     exec(f'self.Rs[2*n_old].{weight}.data += lr*old_model.continuous_block.Rs[n_old].{weight}.data')
      
  #   #     ## (t_H + t_{H+1})/2(t+1)_h --> t_h
  #   #     # if self.interpol == 'constant' or n_old == old_model.N-1:#2:
  #   #       # exec(f'self.Rs[2*n_old + 1].{weight}.data = old_model.continuous_block.Rs[n_old].{weight}.data')
  #   #       # raise Exception('constant interpolation has been removed')
  #   #     # elif self.interpol == 'linear':
  #   #     exec(f'self.Rs[2*n_old + 1].{weight}.data += lr*(' \
  #   #       + f'1/2*(old_model.continuous_block.Rs[n_old].{weight}.data + ' \
  #   #       + f'old_model.continuous_block.Rs[n_old+1].{weight}.data))')
  #   #     # else:
  #   #       # raise Exception('unknown interpolation modality')

  # def restrict_weights_from(self, old_model):    # here old_model is finer
  #   gen_params = old_model.parameters()

  #   for _ in range(1):
  #     _ = next(gen_params)

  #   ## Constant. Later change to linear
  #   for n_coarse in range(self.N):
  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].fc1.weight.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].fc1.bias.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].fc2.weight.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].fc2.bias.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].att.in_proj_weight.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].att.in_proj_bias.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].att.out_proj.weight.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].att.out_proj.bias.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].ln1.weight.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].ln1.bias.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].ln2.weight.data += weights

  #     weights = next(gen_params).data
  #     self.Rs[n_coarse].ln2.bias.data += weights

  #     for _ in range(12):
  #       _ = next(gen_params)

  #   for _ in range(4):
  #     _ = next(gen_params)

  #   assert next(gen_params, None) == None

  #   # # print(f'self.N {self.N} \t old_model.N {old_model.N}')
  #   # for n in range(self.N):    # we undermine the last function weights
  #   #   # print(f'n {n}')
  #   #   for weight in self.weights:
  #   #     ## t_H --> t_h
  #   #     exec(
  #   #       f'self.Rs[n].{weight}.data = 1/2 * (' \
  #   #       + (f'1/2*(old_model.continuous_block.Rs[2*n - 1].{weight}.data.clone()) + ' if n > 0 else '') \
  #   #       + f'old_model.continuous_block.Rs[2*n].{weight}.data.clone()' \
  #   #       + (f' + 1/2*(old_model.continuous_block.Rs[2*n + 1].{weight}.data.clone())' if n < old_model.N else '') \
  #   #       + f')'
  #   #     )

  # def update_diff_weights(self, old_model):
  #   assert old_model.N == self.N

  #   gen_params = old_model.parameters()

  #   for _ in range(1):
  #     _ = next(gen_params)

  #   ## Constant. Later change to linear
  #   for n in range(self.N):
  #     weights = next(gen_params).data
  #     self.Rs[n].fc1.weight.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].fc1.bias.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].fc2.weight.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].fc2.bias.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].att.in_proj_weight.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].att.in_proj_bias.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].att.out_proj.weight.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].att.out_proj.bias.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].ln1.weight.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].ln1.bias.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].ln2.weight.data -= weights

  #     weights = next(gen_params).data
  #     self.Rs[n].ln2.bias.data -= weights

  #   for _ in range(4):
  #     _ = next(gen_params)

  #   assert next(gen_params, None) == None

  #   # for n_old in range(old_model.N):
  #   #   for weight in self.weights:
  #   #     ## t_H --> t_h
  #   #     exec(
  #   #       f'self.Rs[n_old].{weight}.data -= old_model.continuous_block.Rs[n_old].{weight}.data'
  #   #     )
