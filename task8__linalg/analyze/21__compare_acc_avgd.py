import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re

## Colors palette
import seaborn as sns
import colorcet as cc

parser = argparse.ArgumentParser()
parser.add_argument('--fn', type=str, default=None)
parser.add_argument('--contains', type=str, default=None)
parser.add_argument('--avgd', type=int, default=None)
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

def main(contains, fig=None, axs=None, lim=None):
  ## Obtain file name
  if args.fn is not None: fn = args.fn
  else:
    possible_dirs = []
    for dir_nm in os.listdir('../outputs/'):
      if dir_nm.startswith('cont'): possible_dirs.append(dir_nm)

    if len(possible_dirs) == 1: dir_nm = possible_dirs[0]
    else:
      print('Select experiment:')
      for i, dir_nm in enumerate(possible_dirs):
        print(f'  {i} - {dir_nm}')
      x = int(input('--> '))
      dir_nm = possible_dirs[x]

    fns = []
    for fn in os.listdir(f'../outputs/{dir_nm}/outputs/'):
      if fn.startswith('Cont'): fns.append(fn)
  
  if axs is None: fig, axs = plt.subplots(1,2)#3)
  # fig, axs = plt.subplots()
  fig.suptitle(f'{dir_nm}')
  colors = sns.color_palette(cc.glasbey, len(fns))

  fns.sort(key=lambda fn: (-float((fn[fn.find('lr') + len('lr') \
        : fn.find('lr') + fn[fn.find('lr'):].find('_')])),
                           int((fn[fn.find('nlaysenc') + len('nlaysenc') \
        : fn.find('nlaysenc') + fn[fn.find('nlaysenc'):].find('_')]).split('-')[0]),
                           int((fn[fn.find('nlaysdec') + len('nlaysdec') \
       : fn.find('nlaysdec') + fn[fn.find('nlaysdec'):].find('_')]).split('-')[0])))

  for j, fn in enumerate(fns):
    if contains is not None and contains not in fn: continue

    ## Extract accs
    accs_tr, accs_va, changes, times = [], [], [], []
    with open(f'../outputs/{dir_nm}/outputs/{fn}', 'r') as f:
      ctr = 0
      for line in f:
        if line.startswith('training loss'):
          _, acc = line.strip().split('\t')
          acc = float(acc.split()[-1][:-1])
          accs_tr.append(acc)
          ctr += 1

        elif line.startswith('validation loss'):
          _, acc = line.strip().split('\t')
          acc = float(acc.split()[-1][:-1])
          accs_va.append(acc)

        elif line.startswith('Training model w/'):
          changes.append(ctr)

        if line.startswith('time monitor:'):
          time = re.findall('time monitor: (.*)s', line)[0]
          times.append(float(time))

    ## Filter name
    # replace = 'ContTrans_- nnodes-#n ntasksxnode-#tn proc- .txt-' \
    #           + ' nlaysenc-enc nlaysdec-dec'
    replace = 'ContTrans_- nnodes1- ntasksxnode1- procgpu- .txt-' \
              + ' nlaysenc-enc nlaysdec-dec'
    for s in replace.split(): fn = fn.replace(*s.split('-'))
    fn = fn.replace('_', ' ').strip()

    _ = fn.find('enc') + 3
    if len(fn)<=_+1 or fn[_+1] == ' ': fn = fn[:_] + '0' + fn[_:]
    _ = fn.find('dec') + 3
    if len(fn)<=_+1 or fn[_+1] == ' ': fn = fn[:_] + '0' + fn[_:]
    
    if '23lay' in fn: continue

    ## Avg
    if args.avgd is not None:
      aux = []
      for i in range(0, len(accs_va), args.avgd): 
        aux.append(np.mean(accs_va[i:i+args.avgd]))
      accs_va = aux

    ## Plot
    linestyle = '--' if 'conv' in fn else '-'
    axs[0].plot(accs_va, color=colors[j], label=f'{fn}', linestyle=linestyle)
    if lim is not None: axs[0].set_xlim(0, lim[0])
    try: 
      times = np.cumsum(times)
      axs[1].plot(times, accs_va, color=colors[j], label=f'{fn}', 
                                             linestyle=linestyle)
      if lim is not None: axs[1].set_xlim(0, lim[1])
    except: pass
    # axs.plot(accs_va, color=colors[j], label=f'{fn}')
    # ## Gradient computations
    # # n_samples_1monitor = 64*25000#1999998
    # # n_params_1lay = 789760
    # n_layers = (lambda s: int(s[:s.find('l')]))(fn.split()[-1]) 
    # n_grad_comp_1monitor = n_layers#n_samples_1monitor*n_params_1lay*n_layers
    # axs[2].plot(np.arange(1, len(accs_va)+1)*n_grad_comp_1monitor,
    #             accs_va, color=colors[j], label=f'{fn}')

    changes = np.array(changes)
    dv = 1/2*j/len(fns)*2
    axs[0].vlines(changes+dv, 0+dv, 100+dv, color=colors[j], linestyle=':')
    try: axs[1].vlines([times[change]+dv for change in changes], 0+dv, 100+dv,
                                               color=colors[j], linestyle=':')
    except: pass

    try: print(fn, f'max: {max(accs_va)}')
    except: pass

  titles = ['acc_va x 25000 batches', 'acc_va vs. t', 
            'acc_va vs. n_layers (prop. #grad_comps)']
  for (ax, title) in zip(axs, titles): 
    # ax.legend(loc='upper left')
    ax.legend(loc='lower right')
    ax.grid(); ax.set_title(title)
  # axs.legend(loc='upper left'); axs.grid();

  plt.show()


if __name__ == '__main__': 
    if not args.interactive: main(args.contains)
    else:
        fig, axs = plt.subplots(1,2)
        while 1:
            contains = input('Contains: ')
            for ax in axs: ax.clear()
            main(contains, fig, axs)



































