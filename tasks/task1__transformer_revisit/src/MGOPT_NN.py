import numpy as np
import torch

import copy
import tqdm

from itertools import starmap

## Colored output in terminal
# try:
# 	from termcolor import colored
# except:
# 	color = lambda z, col: print(z)
# else:
# 	color = lambda z, col: print(colored(z), col)
color = lambda z, col: print(z)

# Is = []
# Rs = []
# def initIR(models):
# 	for j, model in enumerate(models):
# 		if j == len(models) - 1:
# 			continue

# 		N = model.N - 1

# 		I = np.zeros((2*N + 1, N + 1), dtype=np.float32)
# 		for i in range(N + 1):
# 			I[2*i, i] = 1

# 			if i < N:
# 				I[2*i + 1, i  ] = 1/2
# 				I[2*i + 1, i+1] = 1/2

# 		## Version 1
# 		# R = 1/2 * I.T

# 		## Version 2
# 		R = (I.T).copy()
# 		R /= R.sum(axis=1)[:, None]

# 		print('I', I)
# 		print('R', R)

# 		Is.append(I)
# 		Rs.append(R)

num_params = lambda model: 1 + 12*model.N + 4

def fwd(model, inputs, targets, loss_fn):
	outputs = model(inputs)
	L = loss_fn(outputs, targets)
	return L

def bwd(L, optimizer):
	optimizer.zero_grad()
	L.backward()

def dg_(model_old, model_new, inputs, targets, optimizer_old, optimizer_new, loss_fn):
	## Old model's gradients
	L = fwd(model_old, inputs, targets, loss_fn)
	bwd(L, optimizer_old)
	gen_params_old = model_old.parameters()

	## New model's gradients
	L = fwd(model_new, inputs, targets, loss_fn)
	bwd(L, optimizer_new)
	gen_params_new = model_new.parameters()

	print(len(list(model_old.parameters())), len(list(model_new.parameters())))

	emb_old, emb_new = next(gen_params_old), next(gen_params_new)
	yield(emb_old.data - emb_new.data)

	for n in range(model_new.N):
		## Injection
		for _ in range(12):
			w_old, w_new = next(gen_params_old), next(gen_params_new)
			yield(w_old.data - w_new.data)

		for _ in range(12):
			_ = next(gen_params_old)
		############

	ln3w_old, ln3w_new = next(gen_params_old), next(gen_params_new)
	yield(ln3w_old.data - ln3w_new.data)

	ln3b_old, ln3b_new = next(gen_params_old), next(gen_params_new)
	yield(ln3b_old.data - ln3b_new.data)

	fc3w_old, fc3w_new = next(gen_params_old), next(gen_params_new)
	yield(fc3w_old.data - fc3w_new.data)

	fc3b_old, fc3b_new = next(gen_params_old), next(gen_params_new)
	yield(fc3b_old.data - fc3b_new.data)

	assert next(gen_params_old, None) == None and next(gen_params_new, None) == None


	

# def Rxgrad_(model, inputs, targets, optimizer, loss_fn, dg):
# 	## Fwd
# 	outputs = model(inputs)
# 	L = loss_fn(outputs, targets)

# 	# print(f'weights.device {weights.device}')
# 	# print(f'dg.device {dg.device}')
# 	# print(f'L.device {L.device}')
# 	# H = L.to(device) + (dg.to(device) @ weights.to(device))

# 	## Bwd
# 	optimizer.zero_grad()
# 	L.backward()

# 	if dg != None:
# 		model.add2wgrad(dg)

# 	Rxgrad = model.Rxgrad_()

# 	return Rxgrad

def H_(model, L, dg):
	if dg == None:
		H = L

	else:
		weights = model.parameters()#model.weights_flat_()
		dgxws = sum(starmap(lambda x, y: (x*y).sum(), zip(dg, weights)))
		H = L + dgxws

	return H


def train_step(model, inputs, targets, optimizer, loss_fn, dg, prerestr_loss_hist_i):
	# ## Avoidable if H isn't computed:
	# l_dg = list(dg)
	# dg, dg2 = iter(l_dg), iter(l_dg)

	L = fwd(model, inputs, targets, loss_fn)
	H = H_(model, L, dg)#dg2)

	print(f'\t\t\tL: {L.item() : .2f}, H : {H.item() : .2f} before step')
	# ggg = model.grad_(); print(f'\t\t\tnorm grad (L) = {torch.norm(ggg, 2)} before step' if ggg != None else '\t\t\tnorm grad (L) = None before step')
	# gggdg = model.grad_(); print(f'\t\t\tnorm grad (H) = {torch.norm(ggg + dg, 2)} before step' if ggg != None and dg != None else '\t\t\tnorm grad (H) = None before step')

	bwd(L, optimizer)
	if dg != None:
		model.add2wgrad(dg)
	optimizer.step()

	L = fwd(model, inputs, targets, loss_fn)
	H = H_(model, L, dg)
	bwd(L, optimizer)

	print(f'\t\t\tL: {L.item() : .2f}, H : {H.item() : .2f} after step')
	# ggg = model.grad_(); print(f'\t\t\tnorm grad (L) = {torch.norm(ggg, 2)} after step' if ggg != None else '\t\t\tnorm grad (L) = None after step')
	# gggdg = model.grad_(); print(f'\t\t\tnorm grad (H) = {torch.norm(ggg + dg, 2)} after step' if ggg != None and dg != None else '\t\t\tnorm grad (H) = None after step')
	# if prerestr_loss_hist_i != None:
	# 	prerestr_loss_hist_i[0][prerestr_loss_hist_i[1]] = (L.item(), H.item())


def V_cycle(models, inputs, targets, optimizers, loss_fn, mus_nus, lr_MGOPT, pr=True):
	## Only when weights interp/restr implementation by matrix mult
	# if not(len(Is) == len(Rs) == len(models) - 1):
	# 	raise Exception('Error/Absence of initialization of Is and Rs')

	## Init
	copy_models = [None]*(len(models) - 1)
	dgs = [None]*len(models)
	## finest loss is the conventional loss
	device = next(models[-1].parameters()).device
	# dgs[-1] = torch.zeros(models[-1].num_params, dtype=torch.float32).to(device)

	prerestr_loss_hist = [None]*len(models)

	## From fine to coarse
	for i in range(len(models) - 1, 0, -1):
		model = models[i]
		optimizer = optimizers[i]
		dg = dgs[i]
		mu = mus_nus[i][0]

		## Smooth
		model.train()
		for it in range(mu):
			if pr: color(f'\t\tPresmoothing - Iteration #{it}', 'grey')
			train_step(model, inputs, targets, optimizer, loss_fn, dg, (prerestr_loss_hist, i))

		# Rxgrad = Rxgrad_(model, inputs, targets, optimizer, loss_fn, dg)

		## Restrict
		if pr: color(f'\t\tRestricting model {i} --> {i-1}', 'red')
		model_new = models[i-1]
		model_new.restrict_weights_from(model)
		copy_models[i-1] = copy.deepcopy(model_new)

		# grad_newL = gradL_(model_new, inputs, targets, optimizer, loss_fn)
		# print(f'Rxgrad.shape {Rxgrad.shape}')
		# print(f'grad_newL.shape {grad_newL.shape}')
		# print(f'model_new.num_params {model_new.num_params}')
		# dgs[i-1] = Rxgrad - grad_newL
		# print(f'\t\t\tnorm of Rxgrad    {torch.norm(Rxgrad, 2) : .2f}')
		# print(f'\t\t\tnorm of grad_newL {torch.norm(grad_newL, 2) : .2f}')
		# print(f'\t\t\tnorm of dg        {torch.norm(dgs[i-1], 2) : .2f}')
		optimizer_new = optimizers[i-1]
		dgs[i-1] = dg_(model, model_new, inputs, targets, optimizer, optimizer_new, loss_fn)
		# Sad:
		dgs[i-1] = list(dgs[i-1])

	## Coarsest level
	model = models[0]
	optimizer = optimizers[0]
	dg = dgs[0]
	mu = mus_nus[0]

	for it in range(mu):
		if pr: color(f'\t\tCoarse lvl - Iteration #{it}', 'grey')
		train_step(model, inputs, targets, optimizer, loss_fn, dg, None)

	## From coarse to fine
	for i in range(1, len(models)):
		model, old_model = models[i], models[i-1]

		## Interpolate & correct
		if pr: color(f'\t\tComputing correction #{it}', 'orange')
		old_model.update_diff_weights(copy_models[i-1])

		if pr: color(f'\t\tInterpolating correction', 'green')
		model.interpolate_weights_from(old_model, lr=lr_MGOPT)

		optimizer = optimizers[i]
		dg = dgs[i]
		nu = mus_nus[i][1]

		## Smooth
		for it in range(nu):
			if pr: color(f'\t\tPostsmoothing - Iteration #{it}', 'grey')
			train_step(model, inputs, targets, optimizer, loss_fn, dg, None)
			# print(f'before going coarse:   L: {prerestr_loss_hist[i][0] : .2f}, H: {prerestr_loss_hist[i][1] : .2f}')


































