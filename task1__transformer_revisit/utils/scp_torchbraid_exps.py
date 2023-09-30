import subprocess

def main(t):
	subprocess.call(f'scp -r experiments daint:{t}'.split())

if __name__ == '__main__':
	main()