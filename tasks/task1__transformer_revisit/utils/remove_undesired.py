import os
import shutil
import re

## To remove
patterns = ['.*\.DS_Store', '.*__pycache__', '.*/\._', '\._.*']

def undesired_(l):
	r = re.compile('|'.join(patterns))
	f = filter(r.match, l)
	return f

def do():
	queue_files = sorted(os.listdir())
	while queue_files != []:
		file = queue_files.pop(0)
		if re.match('|'.join(patterns), file):
			if os.path.isdir(file):
				shutil.rmtree(file)
			else:
				os.remove(file)
		elif os.path.isdir(file):
			queue_files += [file + '/' + i for i in sorted(os.listdir(file))]

if __name__ == '__main__':
	do()





