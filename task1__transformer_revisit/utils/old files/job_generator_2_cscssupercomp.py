import os
import re

# if 'experiments' not in os.listdir():
# 	os.mkdir('experiments')

text_ = lambda z, a, b, c, d, e, f, g, h, i: f'''#!/bin/bash -l
#SBATCH --job-name="job{str(z).zfill(2)}"
#SBATCH --account="c24"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marc.salvado@usi.ch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=/users/msalvado/MLT/ML_PQ/outputs/continuous_transformer__20220910_01_tf2dataloaders/ContTrans_lr{a}_init{b}_pe{c}_T{str(d).zfill(2)}_N{e}_numepochs{f}_interpol{g}_batch{h}_lrfactor{i}.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUNPATH=/users/msalvado/MLT/ML_PQ/src
cd $RUNPATH
source activate MLT

srun python3 -u main.py --model continuoustransformer --lr {a} --init {b} --pe {c} --T {d} --N {e} --num_epochs {f} --interpol {g} --batch_size {h} --lr_factor {i} --n_monitoring 10
'''

r = re.compile('exp.*')
ctr = len(list(filter(r.match, os.listdir('experiments'))))
print(f'ctr={ctr}')
for i in range(2, 3):#4):
	lrstr = f'1e-{i}'
	for init in ['Normal']:#['Normal', 'Xavier']:
		for pe in ['Torch']:#['Torch', 'Alternative']:
			for T in [5]:#1, 5, 10]:
				## Experiments 1 & 2 --> 1 is out
				# for N in ['5-9-17-33']:#['5-9-17', '17']:
				# 	for num_epochs in ['1280-1280-1280-1280']:
				# 		for interpol in ['linear']:#['constant', 'linear']:
				# 			for batch_size in [8]:
				# 				for lr_factor in [1.]:
				# 					ctr += 1
				# 					f = open(f'experiments/exp{str(ctr).zfill(2)}.job', 'w')
				# 					text = text_(ctr, lrstr, init, pe, T, N, num_epochs, interpol, batch_size, lr_factor)
				# 					f.write(text)
				# 					f.close()

				## Experiment 3
				# for N in ['33']:
				# 	for num_epochs in [f'{1280*4}']:
				# 		for interpol in ['constant']:
				# 			for batch_size in [8]:
				# 				for lr_factor in [1.]:
				# 					ctr += 1
				# 					f = open(f'experiments/exp{str(ctr).zfill(2)}.job', 'w')
				# 					text = text_(ctr, lrstr, init, pe, T, N, num_epochs, interpol, batch_size, lr_factor)
				# 					f.write(text)
				# 					f.close()

				## Experiment 4
				for N in ['5-9-17-33']:
					for num_epochs in ['1280-640-320-160']:
						for interpol in ['linear']:
							for batch_size in [8]:
								for lr_factor in [1.]:
									ctr += 1
									f = open(f'experiments/exp{str(ctr).zfill(2)}.job', 'w')
									text = text_(ctr, lrstr, init, pe, T, N, num_epochs, interpol, batch_size, lr_factor)
									f.write(text)
									f.close()

				##Experiment 5 & 6 (5.2)
				# for N in ['5-9-17-33']:
				# 	for num_epochs in ['1280-1280-1280-1280']:
				# 		for interpol in ['linear']:
				# 			for batch_size in [8]:
				# 				for lr_factor in [.5, .8]:
				# 					ctr += 1
				# 					f = open(f'experiments/exp{str(ctr).zfill(2)}.job', 'w')
				# 					text = text_(ctr, lrstr, init, pe, T, N, num_epochs, interpol, batch_size, lr_factor)
				# 					f.write(text)
				# 					f.close()








































