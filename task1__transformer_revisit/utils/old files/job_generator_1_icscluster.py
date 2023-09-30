import os

# if 'experiments' not in os.listdir():
# 	os.mkdir('experiments')

text_ = lambda z, a, b, c, d, e, f, g: f'''#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name="job{str(z).zfill(2)}"
#SBATCH --partition=gpu
#SBATCH --output=/home/salvado/MLT/ML_PQ/outputs/continuous_transformer__20220811_01_interpollinear/ContTrans_lr{a}_init{b}_pe{c}_T{str(d).zfill(2)}_N{e}_numepochs{f}_interpol{g}.txt
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive


RUNPATH=/home/salvado/MLT/ML_PQ/src
cd $RUNPATH
source activate MLT


python3 -u main.py --model continuoustransformer --lr {a} --init {b} --pe {c} --T {d} --N {e} --num_epochs {f} --interpol {g}
'''

ctr = 0
for i in range(2, 4):
	lrstr = f'1e-{i}'
	for init in ['Normal']:#['Normal', 'Xavier']:
		for pe in ['Torch']:#['Torch', 'Alternative']:
			for T in [1, 5, 10]:
				for N in ['5-9-17']:#['5-9-17', '17']:
					for num_epochs in [240]:
						for interpol in ['constant', 'linear']:
							ctr += 1
							f = open(f'experiments/exp{str(ctr).zfill(2)}.job', 'w')
							text = text_(ctr, lrstr, init, pe, T, N, num_epochs, interpol)
							f.write(text)
							f.close()
