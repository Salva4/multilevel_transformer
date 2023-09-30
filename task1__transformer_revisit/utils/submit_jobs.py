import os
import sys
import pyautogui as pag
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exclude', type=str, default='debug.job')#None)	# exclude <=
args = parser.parse_args()

sys.path.append('utils')
import remove_undesired

## Parameters #####
path = 'experiments/'
###################

def main():
	morethan = (False, None) if args.exclude == None else (True, args.exclude)

	time.sleep(5)

	listdir = sorted(os.listdir(path))

	for filename in listdir:
		if filename[-4:] == '.job' and \
		  ((not morethan[0]) or (filename[-len(morethan[1]):] > morethan[1])):
			pag.write(f'sbatch {path}{filename}')
			time.sleep(.1)
			pag.press('return')
			time.sleep(.1)

if __name__=='__main__':
    ## Removes files: .DS_Store, __pycache__/*
    remove_undesired.do()
    main()
    remove_undesired.do()




























