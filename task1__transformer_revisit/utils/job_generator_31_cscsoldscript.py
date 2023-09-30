import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ctr', type=int, default=None)
args = parser.parse_args()

# if 'experiments' not in os.listdir():
# 	os.mkdir('experiments')

time_cpu = '10:00:00'
time_gpu = '10:00:00'
partition = 'normal'#'debug'
output_dir = 'continuous_transformer__20221202_03_samebut_Adam'
outsdir_path = '/users/msalvado/MLT/ML_PQ/outputs'

INF = 1000000000

def fill(z, k):
	return str(int(z)).zfill(2)

def text_(**kwargs):
	ctr = kwargs['ctr']
	n_nodes = kwargs['n_nodes']
	n_tasksxnode = kwargs['n_tasksxnode']
	proc = kwargs['proc']
	n_monitoring = kwargs['n_monitoring']
	lr = kwargs['lr']
	optim = kwargs['optim']
	init = kwargs['init']
	pe = kwargs['pe']
	N = kwargs['N']
	T = kwargs['T']
	epochs = kwargs['epochs']
	# interpol = kwargs['interpol']
	batch_size = kwargs['batch_size']
	# lr_factor = kwargs['lr_factor']
	n_lvls = kwargs['n_lvls']
	n_V_cycles = kwargs['n_V_cycles']
	mus_nus = kwargs['mus_nus']
	lr_MGOPT = kwargs['lr_MGOPT']

	## str mods
	musnus_str = mus_nus.replace('_', '--')

	filename = f'ContTrans_nnodes{n_nodes}_ntasksxnode{n_tasksxnode}_proc{proc}_lr{lr}_optim{optim}_init{init}_pe{pe}_N{N}_T{fill(T, 2)}_epochs{epochs}_batchsize{batch_size}_nlvls{n_lvls}_nVcycles{n_V_cycles}_musnus{musnus_str}_lrMGOPT{lr_MGOPT :.2f}'
	output_path = f'{outsdir_path}/{output_dir}/outputs/{filename}.txt'
	error_path = f'{outsdir_path}/{output_dir}/errors/error_flags_{filename}.txt'
	models_path = f'{outsdir_path}/{output_dir}/models'

	t = f'''
#!/bin/bash -l
#SBATCH --job-name="job{str(ctr).zfill(2)}{'_cpu' if proc == 'cpu' else '_gpu'}_alt2"
#SBATCH --account="c24"
#SBATCH --mail-type=ALL
#SBATCH --time={time_cpu if proc == 'cpu' else time_gpu}
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node={n_tasksxnode}
#SBATCH --cpus-per-task=1
#SBATCH --partition={partition}
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --error={error_path}
#SBATCH --output={output_path}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=true

RUNPATH=/users/msalvado/MLT/ML_PQ/src
cd $RUNPATH
source activate MLT_gpu

srun python3 -u main.py --output_fn {filename} --models_dir {models_path} --model continuoustransformer --lr {lr} --optim {optim} --init {init} --pe {pe} --N {N} --T {T} --num_epochs {epochs} --batch_size {batch_size} --n_monitoring {n_monitoring} --n_lvls {n_lvls} --n_V_cycles {n_V_cycles} --mus_nus {mus_nus} --lr_MGOPT {lr_MGOPT}	
'''.strip()
	return t

def main():
	r = re.compile('exp.*')
	ctr = args.ctr if args.ctr != None else len(list(filter(r.match, os.listdir('experiments'))))
	print(f'ctr={ctr}')

	for n_nodes in [1]:
		for n_tasksxnode in [1]:
			for proc in ['gpu']:
				for i in range(2, 3):
					lrstr = f'1e-{i}'
					for optim in ['Adam']:#['SGD']:
						for init in ['xavier']:
							for pe in ['torch']:
								for N in [4]:
									# T = N
									for T in [1., 2., 4.]:
										for epochs in [INF]: 
											for batch_size in [16]:#64 breaks in gpu
												for n_lvls in [2]:
													for n_V_cycles in [1]:
														for mus_nus in ['1_1-1']:
															for lr_MGOPT in [-1., 0., 1e-2, 1e-1, .5, 1.]:
																n_monitoring = 1
																f = open(f'experiments/exp{str(ctr).zfill(2)}.job', 'w')
																text = text_(
																	ctr=ctr,
																	n_nodes=n_nodes,
																	n_tasksxnode=n_tasksxnode,
																	proc=proc,
																	n_monitoring=n_monitoring,
																	lr=lrstr, 
																	optim=optim,
																	init=init,
																	pe=pe,
																	N=N,
																	T=T, 
																	epochs=epochs, 
																	# interpol=interpol,
																	batch_size=batch_size,
																	# lr_factor=lr_factor,
																	n_lvls=n_lvls,
																	n_V_cycles=n_V_cycles,
																	mus_nus=mus_nus,
																	lr_MGOPT=lr_MGOPT,
																)
																f.write(text)
																f.close()
																ctr += 1

	#SBATCH --mail-user=marc.salvado@usi.ch
	#SBATCH --output=/users/msalvado/parallel_in_time/torchbraid/examples/transformer/outputs/{output_dir}/{filename}.txt

	if partition != 'debug':
		f = open(f"experiments/{sorted(os.listdir('experiments'))[0]}", 'r')
		t = f.read()
		f.close()
		f = open('experiments/debug.job', 'w')
		t = t.replace('normal', 'debug')
		t = re.sub('\d\d:\d\d:\d\d', '00:30:00', t)
		f.write(t)
		f.close()

if __name__ == '__main__':
	main()




















