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

def compute_cost(fn, multiplier):
  n_lays_enc = int(re.findall('nlaysenc([^\.\_-]*)', fn)[0])*multiplier
  n_lays_dec = int(re.findall('nlaysdec([^\.\_-]*)', fn)[0])*multiplier
  scheme = re.findall('conv|Euler|Heun|RK4', fn)[0]
  # print(n_lays_enc)

  ## Fixed params
  nparams_emb_enc = 12032
  nparams_emb_dec = 3840
  nparams_fc = 3855
  nparams_enc_layer = 789760
  nparams_dec_layer = 1053440
  n_params_fix = nparams_emb_enc + nparams_emb_dec + nparams_fc
  n_params_var = n_lays_enc*nparams_enc_layer + n_lays_dec*nparams_dec_layer
  # n_params = n_params_fix + n_params_var

  scheme_factor = {   'conv': 1,
                     'Euler': 1,
                      'Heun': 2,
                       'RK4': 2,  }

  n_grad_evals = n_params_fix + n_params_var*scheme_factor[scheme]

  return n_grad_evals


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

  try: fns.sort(key=lambda fn: (-float((fn[fn.find('lr') + len('lr') \
          : fn.find('lr') + fn[fn.find('lr'):].find('_')])),
                             int((fn[fn.find('nlaysenc') + len('nlaysenc') \
          : fn.find('nlaysenc') + fn[fn.find('nlaysenc'):].find('_')]).split('-')[0]),
                             int((fn[fn.find('nlaysdec') + len('nlaysdec') \
          : fn.find('nlaysdec') + fn[fn.find('nlaysdec'):].find('_')]).split('-')[0])))
  except: pass

  for j, fn in enumerate(fns):
    if contains is not None and contains not in fn: continue
    orig_fn = fn
    print(orig_fn)

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

    ## Computational cost
    print(changes)
    if changes == []: changes = [0]
    cost = []
    ctr = 0
    for k in range(len(accs_va)):
      if ctr < len(changes) and changes[ctr] == k: 
        n_grad_evals = compute_cost(orig_fn, multiplier=2**ctr)
        ctr += 1
      cost.append(n_grad_evals)
    # cost = np.array(cost) * np.arange(len(cost))
    cost = np.cumsum(cost)

    print(cost)

    ## Plot
    linestyle = '--' if 'conv' in fn else '-'
    axs[0].plot(accs_va, color=colors[j], label=f'{fn}', linestyle=linestyle)
    if lim is not None: axs[0].set_xlim(0, lim[0])
    axs[1].plot(cost, accs_va, color=colors[j], label=f'{fn}', 
                                          linestyle=linestyle)
    ## Time: currently disabled
    # try: 
    #   times = np.cumsum(times)
    #   axs[1].plot(times, accs_va, color=colors[j], label=f'{fn}', 
    #                                          linestyle=linestyle)
    #   if lim is not None: axs[1].set_xlim(0, lim[1])
    # except: pass

    changes = np.array(changes)
    dv = 1/2*j/len(fns)*2
    axs[0].vlines(changes+dv, 0+dv, 100+dv, color=colors[j], linestyle=':')
    # try: axs[1].vlines([times[change]+dv for change in changes], 0+dv, 100+dv,
    #                                            color=colors[j], linestyle=':')
    # except: pass

    try: print(fn, f'max: {max(accs_va)}')
    except: pass

  titles = ['acc_va x 25000 batches (prop. to #epochs)', 
            'acc_va vs. #gradient computations']
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



































