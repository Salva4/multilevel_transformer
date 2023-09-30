import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy

from models import PositionalEncoding, PE_Alternative

class continuous_block(nn.Module):
	def __init__(self, R, T, N):#, interpol):
		super(continuous_block, self).__init__()
		self.T = T
		self.N = N
		# self.interpol = interpol
		self.dt = T/N
		self.Rs = nn.ModuleList([copy.deepcopy(R) for _ in range(self.N)])

		self.weights = [
			'fc1.weight', 'fc1.bias', 
			'fc2.weight', 'fc2.bias', 
			'att.in_proj_weight', 'att.in_proj_bias', 'att.out_proj.weight', 'att.out_proj.bias', 
			'ln1.weight', 'ln1.bias', 
			'ln2.weight', 'ln2.bias'
		]

	def forward(self, x, **d_extra):
		for n in range(self.N):
			x = x + self.dt*self.Rs[n](x, d_extra)
		return x

	def init_weights_from_model(self, old_model):
		self.interpolate_weights_from(old_model)

	def interpolate_weights_from(self, old_model, lr):	# here old_model is coarser
		gen_params = old_model.parameters()

		for _ in range(1):
			_ = next(gen_params)

		## Constant. Later change to linear
		for n_coarse in range(old_model.N):
			weights = next(gen_params).data
			self.Rs[2*n_coarse].fc1.weight.data += lr * weights
			self.Rs[2*n_coarse + 1].fc1.weight.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].fc1.bias.data += lr * weights
			self.Rs[2*n_coarse + 1].fc1.bias.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].fc2.weight.data += lr * weights
			self.Rs[2*n_coarse + 1].fc2.weight.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].fc2.bias.data += lr * weights
			self.Rs[2*n_coarse + 1].fc2.bias.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].att.in_proj_weight.data += lr * weights
			self.Rs[2*n_coarse + 1].att.in_proj_weight.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].att.in_proj_bias.data += lr * weights
			self.Rs[2*n_coarse + 1].att.in_proj_bias.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].att.out_proj.weight.data += lr * weights
			self.Rs[2*n_coarse + 1].att.out_proj.weight.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].att.out_proj.bias.data += lr * weights
			self.Rs[2*n_coarse + 1].att.out_proj.bias.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].ln1.weight.data += lr * weights
			self.Rs[2*n_coarse + 1].ln1.weight.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].ln1.bias.data += lr * weights
			self.Rs[2*n_coarse + 1].ln1.bias.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].ln2.weight.data += lr * weights
			self.Rs[2*n_coarse + 1].ln2.weight.data += lr * weights

			weights = next(gen_params).data
			self.Rs[2*n_coarse].ln2.bias.data += lr * weights
			self.Rs[2*n_coarse + 1].ln2.bias.data += lr * weights

		for _ in range(4):
			_ = next(gen_params)

		assert next(gen_params, None) == None

		# for n_old in range(old_model.N):	# we undermine the last function weights
		# 	for weight in self.weights:
		# 		## t_H --> t_h
		# 		exec(f'self.Rs[2*n_old].{weight}.data += lr*old_model.continuous_block.Rs[n_old].{weight}.data')
			
		# 		## (t_H + t_{H+1})/2(t+1)_h --> t_h
		# 		# if self.interpol == 'constant' or n_old == old_model.N-1:#2:
		# 			# exec(f'self.Rs[2*n_old + 1].{weight}.data = old_model.continuous_block.Rs[n_old].{weight}.data')
		# 			# raise Exception('constant interpolation has been removed')
		# 		# elif self.interpol == 'linear':
		# 		exec(f'self.Rs[2*n_old + 1].{weight}.data += lr*(' \
		# 			+ f'1/2*(old_model.continuous_block.Rs[n_old].{weight}.data + ' \
		# 			+ f'old_model.continuous_block.Rs[n_old+1].{weight}.data))')
		# 		# else:
		# 			# raise Exception('unknown interpolation modality')

	def restrict_weights_from(self, old_model):		# here old_model is finer
		gen_params = old_model.parameters()

		for _ in range(1):
			_ = next(gen_params)

		## Constant. Later change to linear
		for n_coarse in range(self.N):
			weights = next(gen_params).data
			self.Rs[n_coarse].fc1.weight.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].fc1.bias.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].fc2.weight.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].fc2.bias.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].att.in_proj_weight.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].att.in_proj_bias.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].att.out_proj.weight.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].att.out_proj.bias.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].ln1.weight.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].ln1.bias.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].ln2.weight.data += weights

			weights = next(gen_params).data
			self.Rs[n_coarse].ln2.bias.data += weights

			for _ in range(12):
				_ = next(gen_params)

		for _ in range(4):
			_ = next(gen_params)

		assert next(gen_params, None) == None

		# # print(f'self.N {self.N} \t old_model.N {old_model.N}')
		# for n in range(self.N):		# we undermine the last function weights
		# 	# print(f'n {n}')
		# 	for weight in self.weights:
		# 		## t_H --> t_h
		# 		exec(
		# 			f'self.Rs[n].{weight}.data = 1/2 * (' \
		# 			+ (f'1/2*(old_model.continuous_block.Rs[2*n - 1].{weight}.data.clone()) + ' if n > 0 else '') \
		# 			+ f'old_model.continuous_block.Rs[2*n].{weight}.data.clone()' \
		# 			+ (f' + 1/2*(old_model.continuous_block.Rs[2*n + 1].{weight}.data.clone())' if n < old_model.N else '') \
		# 			+ f')'
		# 		)

	def update_diff_weights(self, old_model):
		assert old_model.N == self.N

		gen_params = old_model.parameters()

		for _ in range(1):
			_ = next(gen_params)

		## Constant. Later change to linear
		for n in range(self.N):
			weights = next(gen_params).data
			self.Rs[n].fc1.weight.data -= weights

			weights = next(gen_params).data
			self.Rs[n].fc1.bias.data -= weights

			weights = next(gen_params).data
			self.Rs[n].fc2.weight.data -= weights

			weights = next(gen_params).data
			self.Rs[n].fc2.bias.data -= weights

			weights = next(gen_params).data
			self.Rs[n].att.in_proj_weight.data -= weights

			weights = next(gen_params).data
			self.Rs[n].att.in_proj_bias.data -= weights

			weights = next(gen_params).data
			self.Rs[n].att.out_proj.weight.data -= weights

			weights = next(gen_params).data
			self.Rs[n].att.out_proj.bias.data -= weights

			weights = next(gen_params).data
			self.Rs[n].ln1.weight.data -= weights

			weights = next(gen_params).data
			self.Rs[n].ln1.bias.data -= weights

			weights = next(gen_params).data
			self.Rs[n].ln2.weight.data -= weights

			weights = next(gen_params).data
			self.Rs[n].ln2.bias.data -= weights

		for _ in range(4):
			_ = next(gen_params)

		assert next(gen_params, None) == None

		# for n_old in range(old_model.N):
		# 	for weight in self.weights:
		# 		## t_H --> t_h
		# 		exec(
		# 			f'self.Rs[n_old].{weight}.data -= old_model.continuous_block.Rs[n_old].{weight}.data'
		# 		)

class dxdtEncoder1DBlock(nn.Module):
	def __init__(self):
		super(dxdtEncoder1DBlock, self).__init__()
		self.fc1 = nn.Linear(128, 128)
		self.fc2 = nn.Linear(128, 128)
		self.att = nn.MultiheadAttention(
			embed_dim=128, 
			num_heads=1, 
			dropout=.3, 
			batch_first=True
		)
		self.ln1 = nn.LayerNorm(128)
		self.ln2 = nn.LayerNorm(128)

	def forward(self, x, d_extra):
		mask = d_extra['mask']

		# ContinuousBlock - dxdtEncoder1DBlock
		x0 = x
		x = self.ln1(x) 		# also try to remove layernorm
		x, _ = self.att(x, x, x, mask)
		x1 = x
		x = x + x0

		x = self.ln2(x)
		# MLPBlock
		x = self.fc1(x)
		x = nn.ELU()(x)
		x = self.fc2(x)
		
		x = x + x1
		return x
		
##
# Continuous transformer encoder layer using their code's scheme & <i>MultiheadAttention</i>
class Continuoustransformer(nn.Module):
	def __init__(self, init_method, encoding, **kwargs):
		print('Model: Continuous transformer')
		super(Continuoustransformer, self).__init__()
		self.init_method, self.encoding = init_method, encoding
		self.T = kwargs['T']
		self.N = kwargs['N']
		# self.interpol = kwargs['interpol']
		print(f'\tinit method: {self.init_method}')
		print(f'\tencoding {encoding}')
		print(f'\tT={self.T}, N={self.N}\n')

		self.emb = nn.Embedding(15514, 128)
		self.dropout = nn.Dropout(p=.1)
		self.posenc = PositionalEncoding(128) if encoding == 'Torch'\
		  else PE_Alternative(128) if encoding == 'Alternative'\
		  else Exception('encoding unknown')

		dxdt = dxdtEncoder1DBlock()
		self.continuous_block = continuous_block(
			R=dxdt, 
			T=self.T,
			N=self.N,
			# interpol=self.interpol,
		)

		self.ln3 = nn.LayerNorm(128)
		self.fc3 = nn.Linear(128, 49)

		if self.init_method != 'None':
			self.init_params()

		self.num_params = self.weights_flat_().shape[0]

	def forward(self, x):
		mask = (x == 0)
		x = self.fwd_pre_contblock(x)
		x = self.continuous_block(x, mask=mask)
		x = self.fwd_post_contblock(x)
		return x

	def fwd_pre_contblock(self, x):
		x = self.emb(x) 
		x = self.dropout(x)
		x = self.posenc(x)
		return x

	def fwd_post_contblock(self, x):
		x = self.ln3(x)
		x = self.fc3(x)
		return x
		
	def init_params(self):
		self.apply(self._init_layers)

	def _init_layers(self, m):
		classname = m.__class__.__name__
		if isinstance(m, nn.Conv2d):
			if m.weight is not None:
				torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

			if m.bias is not None:
				torch.nn.init.constant_(m.bias, 0)
		
		if isinstance(m, nn.BatchNorm2d):
			if m.weight is not None:
				torch.nn.init.constant_(m.weight, 1)

			if m.bias is not None:
				torch.nn.init.constant_(m.bias, 0)
		
		if isinstance(m, nn.Linear):
			if m.weight is not None:
				torch.nn.init.normal_(m.weight)

			if m.bias is not None:
				torch.nn.init.constant_(m.bias, 0)
		
		if isinstance(m, nn.MultiheadAttention):
			if self.init_method == 'Normal':
				m.in_proj_weight.data.normal_(mean=0.0, std=0.02)
				m.out_proj.weight.data.normal_(mean=0.0, std=0.02)
			elif self.init_method == 'Xavier':
				torch.nn.init.xavier_uniform_(m.in_proj_weight, gain=np.sqrt(2))
				torch.nn.init.xavier_uniform_(m.out_proj.weight, gain=np.sqrt(2))
			else:
				Exception('Initialization method unknown')

	# def init_weights_from_model(self, old_model): 		<-- not used in MG/OPT
	# 	self.emb.weight.data = old_model.emb.weight.data
	# 	self.continuous_block.init_weights_from_model(old_model)
	# 	self.ln3.weight.data = old_model.ln3.weight.data
	# 	self.ln3.bias.data = old_model.ln3.bias.data
	# 	self.fc3.weight.data = old_model.fc3.weight.data
	# 	self.fc3.bias.data = old_model.fc3.bias.data



	def interpolate_weights_from(self, old_model, lr):
		## Pre
		self.emb.weight.data += lr*old_model.emb.weight.data

		## Cont
		self.continuous_block.interpolate_weights_from(old_model, lr)

		## Post
		self.ln3.weight.data += lr*old_model.ln3.weight.data
		self.ln3.bias.data += lr*old_model.ln3.bias.data
		self.fc3.weight.data += lr*old_model.fc3.weight.data
		self.fc3.bias.data += lr*old_model.fc3.bias.data

	def restrict_weights_from(self, old_model):
		## Pre
		self.emb.weight.data = old_model.emb.weight.data.clone()

		## Cont
		self.continuous_block.restrict_weights_from(old_model)

		## Post
		self.ln3.weight.data = old_model.ln3.weight.data.clone()
		self.ln3.bias.data = old_model.ln3.bias.data.clone()
		self.fc3.weight.data = old_model.fc3.weight.data.clone()
		self.fc3.bias.data = old_model.fc3.bias.data.clone()

	def update_diff_weights(self, old_model):
		## Pre
		self.emb.weight.data -= old_model.emb.weight.data

		## Cont
		self.continuous_block.update_diff_weights(old_model)

		## Post
		self.ln3.weight.data -= old_model.ln3.weight.data
		self.ln3.bias.data -= old_model.ln3.bias.data
		self.fc3.weight.data -= old_model.fc3.weight.data
		self.fc3.bias.data -= old_model.fc3.bias.data

	def weights_flat_(self):
		weights = torch.cat((self.emb.weight.data.flatten(),
			 				 torch.cat([torch.cat([self.continuous_block.Rs[n].fc1.weight.data.flatten(),
										 self.continuous_block.Rs[n].fc1.bias.data.flatten(),
										 self.continuous_block.Rs[n].fc2.weight.data.flatten(),
										 self.continuous_block.Rs[n].fc2.bias.data.flatten(),
										 self.continuous_block.Rs[n].att.in_proj_weight.data.flatten(),
										 self.continuous_block.Rs[n].att.in_proj_bias.data.flatten(),
										 self.continuous_block.Rs[n].att.out_proj.weight.data.flatten(),
										 self.continuous_block.Rs[n].att.out_proj.bias.data.flatten(),
										 self.continuous_block.Rs[n].ln1.weight.data.flatten(),
										 self.continuous_block.Rs[n].ln1.bias.data.flatten(),
										 self.continuous_block.Rs[n].ln2.weight.data.flatten(),
										 self.continuous_block.Rs[n].ln2.bias.data.flatten()], axis=0)
										 for n in range(self.N)], axis=0),
			 				 self.ln3.weight.data.flatten(),
			 				 self.ln3.bias.data.flatten(),
			 				 self.fc3.weight.data.flatten(),
			 				 self.fc3.bias.data.flatten()), 
			                axis=0)
		return weights

	def grad_(self):
		if self.emb.weight.grad == None:
			return None
			
		grad = torch.cat((self.emb.weight.grad.flatten(),
			 				 torch.cat([torch.cat([self.continuous_block.Rs[n].fc1.weight.grad.flatten(),
										 self.continuous_block.Rs[n].fc1.bias.grad.flatten(),
										 self.continuous_block.Rs[n].fc2.weight.grad.flatten(),
										 self.continuous_block.Rs[n].fc2.bias.grad.flatten(),
										 self.continuous_block.Rs[n].att.in_proj_weight.grad.flatten(),
										 self.continuous_block.Rs[n].att.in_proj_bias.grad.flatten(),
										 self.continuous_block.Rs[n].att.out_proj.weight.grad.flatten(),
										 self.continuous_block.Rs[n].att.out_proj.bias.grad.flatten(),
										 self.continuous_block.Rs[n].ln1.weight.grad.flatten(),
										 self.continuous_block.Rs[n].ln1.bias.grad.flatten(),
										 self.continuous_block.Rs[n].ln2.weight.grad.flatten(),
										 self.continuous_block.Rs[n].ln2.bias.grad.flatten()], axis=0)
										 for n in range(self.N)], axis=0),
			 				 self.ln3.weight.grad.flatten(),
			 				 self.ln3.bias.grad.flatten(),
			 				 self.fc3.weight.grad.flatten(),
			 				 self.fc3.bias.grad.flatten()), 
			                axis=0)
		return grad

	def Rxgrad_(self):
		# cont_grads = [[ self.continuous_block.Rs[n].fc1.weight.grad,
		# 				self.continuous_block.Rs[n].fc1.bias.grad,
		# 				self.continuous_block.Rs[n].fc2.weight.grad,
		# 				self.continuous_block.Rs[n].fc2.bias.grad,
		# 				self.continuous_block.Rs[n].att.in_proj_weight.grad,
		# 				self.continuous_block.Rs[n].att.in_proj_bias.grad,
		# 				self.continuous_block.Rs[n].att.out_proj.weight.grad,
		# 				self.continuous_block.Rs[n].att.out_proj.bias.grad,
		# 				self.continuous_block.Rs[n].ln1.weight.grad,
		# 				self.continuous_block.Rs[n].ln1.bias.grad,
		# 				self.continuous_block.Rs[n].ln2.weight.grad,
		# 				self.continuous_block.Rs[n].ln2.bias.grad
		# 				] for n in range(self.N)]

		# Rxgrad = [
		# 	self.emb.weight.grad,
		# 	[
		# 		[
		# 			1/2*(
		# 					(1/2*cont_grads[2*n - 1][iw] if n > 0 else 0) +\
		# 					cont_grads[2*n][iw] +\
		# 					1/2*cont_grads[2*n + 1][iw]
		# 			) for iw in range(len(cont_grads[0]))
		# 		] for n in range(self.N//2)
		# 	],
		# 	self.ln3.weight.grad,
		# 	self.ln3.bias.grad,
		# 	self.fc3.weight.grad,
		# 	self.fc3.bias.grad, 
		# ]
		cont_grads = [[self.continuous_block.Rs[n].fc1.weight.grad.flatten(),
						self.continuous_block.Rs[n].fc1.bias.grad.flatten(),
						self.continuous_block.Rs[n].fc2.weight.grad.flatten(),
						self.continuous_block.Rs[n].fc2.bias.grad.flatten(),
						self.continuous_block.Rs[n].att.in_proj_weight.grad.flatten(),
						self.continuous_block.Rs[n].att.in_proj_bias.grad.flatten(),
						self.continuous_block.Rs[n].att.out_proj.weight.grad.flatten(),
						self.continuous_block.Rs[n].att.out_proj.bias.grad.flatten(),
						self.continuous_block.Rs[n].ln1.weight.grad.flatten(),
						self.continuous_block.Rs[n].ln1.bias.grad.flatten(),
						self.continuous_block.Rs[n].ln2.weight.grad.flatten(),
						self.continuous_block.Rs[n].ln2.bias.grad.flatten()]
						for n in range(self.N)]

		#### No injection:
		# Rxgrad = torch.cat((self.emb.weight.grad.flatten(),
		# 					torch.cat([torch.cat([1/2*(1/2*cont_grads[2*n - 1][iw] if n > 0 else 0) +\
		# 											cont_grads[2*n][iw] +\
		# 											1/2*cont_grads[2*n + 1][iw]
		# 											for iw in range(len(cont_grads[0]))], axis=0)
		# 							   for n in range(self.N//2)], axis=0),
		# 	 				self.ln3.weight.grad.flatten(),
		# 	 				self.ln3.bias.grad.flatten(),
		# 	 				self.fc3.weight.grad.flatten(),
		# 	 				self.fc3.bias.grad.flatten()), 
		# 	            axis=0)
		## Yes injection:
		Rxgrad = torch.cat((self.emb.weight.grad.flatten(),
							torch.cat([torch.cat([cont_grads[2*n][iw]
												  for iw in range(len(cont_grads[0]))], axis=0)
									   for n in range(self.N//2)], axis=0),
			 				self.ln3.weight.grad.flatten(),
			 				self.ln3.bias.grad.flatten(),
			 				self.fc3.weight.grad.flatten(),
			 				self.fc3.bias.grad.flatten()), 
			            axis=0)
		#### 
		return Rxgrad


	def add2wgrad(self, dg):
		# idx = 0
		
		# self.emb.weight.grad.data += dg[idx : idx + 15514*128].reshape(15514, 128)
		# idx += 15514*128

		# for n in range(self.continuous_block.N):
		# 	self.continuous_block.Rs[n].fc1.weight.grad.data += dg[idx : idx + 128*128].reshape(128, 128)
		# 	idx += 128*128
		# 	self.continuous_block.Rs[n].fc1.bias.grad.data += dg[idx : idx + 128]
		# 	idx += 128
		# 	self.continuous_block.Rs[n].fc2.weight.grad.data += dg[idx : idx + 128*128].reshape(128, 128)
		# 	idx += 128*128
		# 	self.continuous_block.Rs[n].fc2.bias.grad.data += dg[idx : idx + 128]
		# 	idx += 128
		# 	self.continuous_block.Rs[n].att.in_proj_weight.grad.data += dg[idx: idx + 384*128].reshape(384, 128)
		# 	idx += 384*128
		# 	self.continuous_block.Rs[n].att.in_proj_bias.grad.data += dg[idx: idx + 384]
		# 	idx += 384
		# 	self.continuous_block.Rs[n].att.out_proj.weight.grad.data += dg[idx: idx + 128*128].reshape(128, 128)
		# 	idx += 128*128
		# 	self.continuous_block.Rs[n].att.out_proj.bias.grad.data += dg[idx: idx + 128]
		# 	idx += 128
		# 	self.continuous_block.Rs[n].ln1.weight.grad.data += dg[idx: idx + 128]
		# 	idx += 128
		# 	self.continuous_block.Rs[n].ln1.bias.grad.data += dg[idx: idx + 128]
		# 	idx += 128
		# 	self.continuous_block.Rs[n].ln2.weight.grad.data += dg[idx: idx + 128]
		# 	idx += 128
		# 	self.continuous_block.Rs[n].ln2.bias.grad.data += dg[idx: idx + 128]
		# 	idx += 128

		# self.ln3.weight.grad.data += dg[idx: idx + 128]
		# idx += 128
		# self.ln3.bias.grad.data += dg[idx: idx + 128]
		# idx += 128
		# self.fc3.weight.grad.data += dg[idx: idx + 49*128].reshape(49, 128)
		# idx += 49*128
		# self.fc3.bias.grad.data += dg[idx: idx + 49]
		# idx += 49
		# assert idx == dg.shape[0] and dg.ndim == 1

		## Sad:
		if type(dg) == list:
			dg = iter(dg)

		self.emb.weight.grad.data += next(dg)

		for n in range(self.continuous_block.N):
			self.continuous_block.Rs[n].fc1.weight.grad.data += next(dg)
			self.continuous_block.Rs[n].fc1.bias.grad.data += next(dg)
			self.continuous_block.Rs[n].fc2.weight.grad.data += next(dg)
			self.continuous_block.Rs[n].fc2.bias.grad.data += next(dg)
			self.continuous_block.Rs[n].att.in_proj_weight.grad.data += next(dg)
			self.continuous_block.Rs[n].att.in_proj_bias.grad.data += next(dg)
			self.continuous_block.Rs[n].att.out_proj.weight.grad.data += next(dg)
			self.continuous_block.Rs[n].att.out_proj.bias.grad.data += next(dg)
			self.continuous_block.Rs[n].ln1.weight.grad.data += next(dg)
			self.continuous_block.Rs[n].ln1.bias.grad.data += next(dg)
			self.continuous_block.Rs[n].ln2.weight.grad.data += next(dg)
			self.continuous_block.Rs[n].ln2.bias.grad.data += next(dg)

		self.ln3.weight.grad.data += next(dg)
		self.ln3.bias.grad.data += next(dg)
		self.fc3.weight.grad.data += next(dg)
		self.fc3.bias.grad.data += next(dg)

		assert next(dg, None) == None





































