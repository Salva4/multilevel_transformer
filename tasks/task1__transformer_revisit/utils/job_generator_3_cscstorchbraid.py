import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ctr', type=int, default=None)
args = parser.parse_args()

# if 'experiments' not in os.listdir():
# 	os.mkdir('experiments')

time_cpu = '10:00:00'
time_gpu = '24:00:00'	# must have 2 digits for h, m & s! XX:XX:XX
partition = 'normal'#'debug'
output_dir = 'continuous_transformer__20221027_01_convergenceattempt_parallel'
outsdir_path = '/users/msalvado/parallel_in_time/torchbraid/examples/transformer/outputs'

def fill(z, k):
	return str(int(z)).zfill(2)

def text_(**kwargs):
	ctr = kwargs['ctr']
	lr = kwargs['lr']
	epochs = kwargs['epochs']
	tf = kwargs['tf']
	usedowncycle = kwargs['usedowncycle']
	steps = kwargs['steps']
	cfactor = kwargs['cfactor']
	levels = kwargs['levels']
	n_nodes = kwargs['n_nodes']
	n_tasksxnode = kwargs['n_tasksxnode']
	proc = kwargs['proc']
	batch_size = kwargs['batch_size']
	max_iters = kwargs['max_iters']
	filename = f'ContTrans_lr{lr}_epochs{epochs}_tf{fill(tf, 2)}_usedowncycle{usedowncycle}_steps{steps}_cfactor{cfactor}_levels{levels}_nnodes{n_nodes}_ntasksxnode{n_tasksxnode}_proc{proc}_batchsize{batch_size}_maxiters{max_iters}'
	output_path = f'{outsdir_path}/{output_dir}/outputs/{filename}.txt'
	error_path = f'{outsdir_path}/{output_dir}/errors/error_flags_{filename}.txt'
	models_path = f'{outsdir_path}/{output_dir}/models'
	t = f'''
#!/bin/bash -l
#SBATCH --job-name="job{str(ctr).zfill(2)}{'_cpu' if proc == 'cpu' else '_gpu'}"
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

LOADPATH=/users/msalvado/parallel_in_time/
RUNPATH=/users/msalvado/parallel_in_time/torchbraid/examples/transformer/src/
cd $LOADPATH
source modules_pint{'_cpu' if proc == 'cpu' else ''}.sh
cd $RUNPATH

srun python3 -u main.py --lr {lr} --epochs {epochs} --tf {tf} {'--lp-use-downcycle' if usedowncycle else ''} --steps {steps} --lp-cfactor {cfactor} --lp-levels {levels} --batch-size {batch_size} --lp-iters {max_iters} --output_fn {filename} --models_dir {models_path}
'''.strip()
	return t

def main():
	r = re.compile('exp.*')
	ctr = args.ctr if args.ctr != None else len(list(filter(r.match, os.listdir('experiments'))))
	print(f'ctr={ctr}')
	for i in range(4, 5):
		lrstr = f'1e-{i}'
		for epochs in [12*10]:#[1, 10, 100]:
			for tf in [10]:#['same']:
				for usedowncycle in [True]:#[True, False]:
					for steps in [20]:#[20]:#[32, 64, 128]:#[20]
						for cfactor in [4]:#[2, 4]:
							for levels in [3]:#[2, 3]:
								for n_nodes in [2]:#[2, 4, 8]:
									for n_tasksxnode in [1]:#[1, 4]:#[1, 2, 4, 8, 12]:
										for proc in ['gpu']:#['cpu', 'gpu']:
											for batch_size in [16]:#[16, 64]:#[4, 8]:
												for max_iters in [2]:#, 3, 4]:
													tf = steps if tf == 'same' else tf
													f = open(f'experiments/exp{str(ctr).zfill(2)}.job', 'w')
													text = text_(
														ctr=ctr, 
														lr=lrstr, 
														epochs=epochs, 
														tf=tf, 
														usedowncycle=usedowncycle, 
														steps=steps, 
														cfactor=cfactor, 
														levels=levels,
														n_nodes=n_nodes,
														n_tasksxnode=n_tasksxnode,
														proc=proc,
														batch_size=batch_size,
														max_iters=max_iters,
													)
													f.write(text)
													f.close()
													ctr += 1

	#SBATCH --mail-user=marc.salvado@usi.ch

	f = open(f"experiments/{sorted(os.listdir('experiments'))[0]}", 'r')
	t = f.read()
	f.close()
	f = open('experiments/debug.job', 'w')
	t = t.replace('normal', 'debug')	# partition
	t = re.sub('\d\d:\d\d:\d\d', '00:30:00', t) # time for debug
	## add "debug" to job name
	subtext = re.findall('--job-name=[^\n]*', t)[0]
	t = re.sub(subtext, subtext[:-1] + '_debug"', t)
	##
	f.write(t)
	f.close()

if __name__ == '__main__':
	main()




















