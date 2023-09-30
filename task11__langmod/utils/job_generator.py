import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ctr', type=int, default=None)
args = parser.parse_args()

time_cpu = '10:00:00'
time_gpu = '24:00:00'	# must have 2 digits for h, m & s! XX:XX:XX
partition = 'normal'#'debug'
output_dir = 'continuous_transformer__20230404_01_task11_compare'
task_path = '/users/msalvado/prova_transfpytorch/task11__langmod'
outsdir_path = f'{task_path}/outputs'

def fill(z, k):
	return str(int(z)).zfill(2)

def text_(**kwargs):
	ctr = kwargs['ctr']
	n_nodes = kwargs['n_nodes']
	n_tasksxnode = kwargs['n_tasksxnode']
	proc = kwargs['proc']
	num_epochs = kwargs['num_epochs']
	num_layers = kwargs['num_layers']

	filename = f'ContTrans_nnodes{n_nodes}_ntasksxnode{n_tasksxnode}_proc{proc}_nepochs{num_epochs}_nlayers{num_layers}'
	output_path = f'{outsdir_path}/{output_dir}/outputs/{filename}.txt'
	error_path = f'{outsdir_path}/{output_dir}/errors/error_flags_{filename}.txt'
	models_path = f'{outsdir_path}/{output_dir}/models'

	t = f'''
#!/bin/bash -l
#SBATCH --job-name="job{str(ctr).zfill(2)}{'_cpu' if proc == 'cpu' else '_gpu'}_conv_{task_path.split('/')[-1]}"
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

RUNPATH={task_path}/src/
conda activate env1
cd $RUNPATH

srun python3 -u main_MLInit.py --num_epochs {num_epochs} --num_layers {num_layers}
'''.strip()
	return t

def main():
	r = re.compile('exp.*')
	ctr = args.ctr if args.ctr != None else len(list(filter(r.match, os.listdir('../experiments'))))
	print(f'ctr={ctr}')

	for n_nodes in [1]:#, 4, 8]:
		for n_tasksxnode in [1]:#[1, 4]:#[1, 2, 4, 8, 12]:
			for proc in ['gpu']:#['cpu', 'gpu']:
				for num_epochs, num_layers in zip(
																	 ['5000-5000-5000-5000', '20000', '20000'],
																										 ['2-4-8-16', '4', '16']):
						f = open(f'../experiments/exp{str(ctr).zfill(2)}.job', 'w')
						text = text_(
							ctr=ctr, 
							n_nodes=n_nodes,
							n_tasksxnode=n_tasksxnode,
							proc=proc,
							num_epochs=num_epochs,
							num_layers=num_layers,							
						)
						f.write(text)
						f.close()
						ctr += 1

	#SBATCH --mail-user=marc.salvado@usi.ch

	if partition != 'debug':
		f = open(f"../experiments/{sorted(os.listdir('../experiments'))[0]}", 'r')
		t = f.read()
		f.close()
		f = open('../experiments/debug.job', 'w')
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




















