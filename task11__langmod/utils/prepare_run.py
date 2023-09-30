import os
import shutil
import subprocess

import job_generator as jg

def main():
  os.mkdir(f'../outputs/{jg.output_dir}')
  os.mkdir(f'../outputs/{jg.output_dir}/outputs')
  os.mkdir(f'../outputs/{jg.output_dir}/models')
  os.mkdir(f'../outputs/{jg.output_dir}/errors')
  shutil.copytree(f'../src', f'../outputs/{jg.output_dir}/src')

  shutil.rmtree('../experiments')
  os.mkdir(f'../experiments')

  subprocess.call('python job_generator.py'.split())

  shutil.copytree(f'../experiments', 
                  f'../outputs/{jg.output_dir}/experiments')

if __name__ == '__main__':
  main()




















