import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/marcsalvado/Dropbox/Marc/2020-2021/Scripts_python/53__Canva_colorwheel')
import main_palette_for_import as palette

OUTPUTS_PATH = os.path.join('..', 'outputs')

def filter_fn(fn):
  prefix = 'ContTrans_'
  if fn.startswith(prefix): fn = fn[len(prefix):]
  sufix = '.txt'
  if fn.endswith(sufix): fn = fn[:-len(sufix)]
  fn = fn.replace('nnodes', 'nn')
  fn = fn.replace('ntasksxnode', 'ntn')
  fn = fn.replace('proc', '')
  fn = fn.replace('batchsize', 'btchsz')
  return fn

def find_data_path(outputs_path):
  outputs_dirs = os.listdir(outputs_path)
  found = False
  for outputs_dir in outputs_dirs:
    if outputs_dir.startswith('c'):
      found = True
      break
  assert found, '''There is no outputs_directory starting with 'c'.'''
  data_path = os.path.join(outputs_path, outputs_dir, 'outputs')
  return data_path

def main():
  data_path = find_data_path(OUTPUTS_PATH)
  plot_data(data_path)

def plot_data(data_path):
  fns = os.listdir(data_path)
  fns = [fn for fn in fns if fn.startswith('Cont')]
  colors = palette.main(len(fns))
  fig, ax = plt.subplots()
  for j, fn in enumerate(fns):
    full_fn = os.path.join(data_path, fn)
    with open(full_fn, 'r') as f:
      va_accs = []
      for line in f:
        line = line.strip()
        if line.startswith('Epoch'): 
        # if line.startswith('Test'): 
          va_accs.append(float(line.split()[-1])*100)
          # va_accs.append(float(line.split()[-1][1:-2])*100)
    ax.plot(va_accs, linestyle='-', color=colors[j], label=filter_fn(fn)+'_non-tb')
    # ax.plot(va_accs, linestyle='--', color=colors[j], label=filter_fn(fn)+'tb')
  ax.grid()
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Acc')
  ax.set_yticks(range(0, 110, 10))
  ax.legend()
  plt.show()    


if __name__ == '__main__':
  main()











