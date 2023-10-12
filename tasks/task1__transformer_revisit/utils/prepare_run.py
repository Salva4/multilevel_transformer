import pyautogui as pag
import time
import scp_torchbraid_exps as ste

## No es pot!
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--dt', type=float, default=1.)
# args = parser.parse_args()
# dt = args.dt

## Params USER
job_gen = 'job_generator_31_cscsoldscript'#'job_generator_3_cscstorchbraid'
dt = .1
##

sleep = lambda: time.sleep(dt)

if job_gen == 'job_generator_3_cscstorchbraid':
	from job_generator_3_cscstorchbraid import output_dir
elif job_gen == 'job_generator_31_cscsoldscript':
	from job_generator_31_cscsoldscript import output_dir
else:
	print('error')
	exit(0)

def change_tab(z):
	hotkeys = (('Shift',) if z < 0 else ()) + ('Ctrl', 'Tab')
	for i in range(abs(z)):
		pag.hotkey(*hotkeys)
		sleep()

def clear_screen():
	pag.hotkey('Ctrl', 'l')
	sleep()

def cp_exp_outputdir():
	exp_dir = 'experiments'
	pag.write(f'cp -r {exp_dir} outputs/{output_dir}')
	enter()

def cp_src_outputdir():
	src_dir = 'src' if job_gen == 'job_generator_3_cscstorchbraid' else '/users/msalvado/MLT/ML_PQ/src'
	pag.write(f'cp -r {src_dir} outputs/{output_dir}')
	enter()

def enter():
	pag.press('return')	
	sleep()

def follow_debug():
	pag.write('tail -f outputs/c')
	sleep()
	pag.press('Tab')
	sleep()
	pag.press('C')
	time.sleep(3)
	pag.press('Tab')
	time.sleep(1)
	pag.enter()

def mkdir_outputdir():
	pag.write(f'mkdir outputs/{output_dir}')
	enter()

def mkdir_outserr():
	pag.write(f'mkdir outputs/{output_dir}/errors')
	enter()

def mkdir_outsmod():
	pag.write(f'mkdir outputs/{output_dir}/models')
	enter()

def mkdir_outsouts():
	pag.write(f'mkdir outputs/{output_dir}/outputs')
	enter()

def mv_old_outputs():
	pag.write('mv outputs/c* outputs/old_models')
	enter()

def practice_hotkeys():
	for _ in range(3):
		pag.hotkey('Return')
		sleep()

def run_job_gen():
	pag.write(f'python utils/{job_gen}.py')
	enter()

def rm_exps(local):
	pag.write('rm experiments/*')
	enter()
	if local == True:
		pag.write('y')
		sleep()

def sbatch_debug():
	pag.write('sbatch experiments/debug.job')
	enter()

def scp_exps():
	t = 'parallel_in_time/torchbraid/examples/transformer/' if job_gen == 'job_generator_3_cscstorchbraid' else '/users/msalvado/MLT/ML_PQ'
	ste.main(t)
	sleep()

## MAIN
def main():
	time.sleep(1)
	practice_hotkeys()
	change_tab(1)
	rm_exps(local=True)
	change_tab(1)
	rm_exps(local=False)
	# mv_old_outputs()
	mkdir_outputdir()
	cp_src_outputdir()
	mkdir_outsouts()
	mkdir_outserr()
	mkdir_outsmod()
	change_tab(-1)
	run_job_gen()
	change_tab(-1)
	scp_exps()
	clear_screen()
	change_tab(1)
	clear_screen()
	change_tab(1)
	cp_exp_outputdir()
	clear_screen()
	# sbatch_debug()
	# follow_debug()

if __name__ == '__main__':
	main()






































