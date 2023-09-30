import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

## Colors palette
import seaborn as sns
import colorcet as cc

parser = argparse.ArgumentParser()
parser.add_argument('--fn', type=str, default=None)
parser.add_argument('--lr', type=str, default=None)
parser.add_argument('--enc', type=str, default=None)
parser.add_argument('--dec', type=str, default=None)
args = parser.parse_args()

dir_outputs = '3__files'

def main():
  ## Obtain file name
  fns = []
  for fn in os.listdir(f'{dir_outputs}/'):
    if fn.startswith('Cont'): fns.append(fn)

  fig, axs = plt.subplots(1,2)#3)
  # fig, axs = plt.subplots()
  colors = sns.color_palette(cc.glasbey, len(fns))

  # fns.sort(key=lambda fn: (-float((fn[fn.find('lr') + len('lr') \
  #       : fn.find('lr') + len('lr') + 5])),
  #                          int((fn[fn.find('nlaysenc') + len('nlaysenc') \
  #       : fn.find('nlaysenc') + len('nlaysenc') + 2]).replace('_',' ')),
  #                          int((fn[fn.find('nlaysdec') + len('nlaysdec') \
  #      : fn.find('nlaysdec') + len('nlaysdec') + 2]).replace('.',' '))))

  for j, fn in enumerate(fns):
    if args.lr is not None and args.lr not in fn: continue
    if args.enc is not None and args.enc not in fn: continue
    if args.dec is not None and args.dec not in fn: continue

    ## Extract accs
    accs_tr, accs_va = [], []
    with open(f'{dir_outputs}/{fn}', 'r') as f:
      for line in f:
        if line.startswith('training loss'):
          _, acc = line.strip().split('\t')
          acc = float(acc.split()[-1][:-1])
          accs_tr.append(acc)

        if line.startswith('validation loss'):
          _, acc = line.strip().split('\t')
          acc = float(acc.split()[-1][:-1])
          accs_va.append(acc)

    ## Filter name
    replace = 'ContTrans_- nnodes-#n ntasksxnode-#tn proc- .txt-' \
              + ' nlaysenc-enc nlaysdec-dec'
    for s in replace.split(): fn = fn.replace(*s.split('-'))
    fn = fn.replace('_', ' ').strip()

    _ = fn.find('enc') + 3
    if len(fn)<=_+1 or fn[_+1] == ' ': fn = fn[:_] + '0' + fn[_:]
    _ = fn.find('dec') + 3
    if len(fn)<=_+1 or fn[_+1] == ' ': fn = fn[:_] + '0' + fn[_:]
    
    if '23lay' in fn: continue

    # aux = []
    # for i in range(0, len(accs_va), 5): 
    #   aux.append(np.mean(accs_va[i:i+5]))
    # accs_va = aux

    ## Plot
    axs[0].plot(accs_va, color=colors[j], label=f'{fn}')
    axs[1].plot(np.arange(len(accs_va))/len(accs_va),
                accs_va, color=colors[j], label=f'{fn}')
    # axs.plot(accs_va, color=colors[j], label=f'{fn}')
    # ## Gradient computations
    # # n_samples_1monitor = 64*25000#1999998
    # # n_params_1lay = 789760
    # n_layers = (lambda s: int(s[:s.find('l')]))(fn.split()[-1]) 
    # n_grad_comp_1monitor = n_layers#n_samples_1monitor*n_params_1lay*n_layers
    # axs[2].plot(np.arange(1, len(accs_va)+1)*n_grad_comp_1monitor,
    #             accs_va, color=colors[j], label=f'{fn}')

    try: print(fn, f'max: {max(accs_va)}')
    except: pass

  titles = ['acc_va x 25000 batches', 'acc_va vs. t', 
            'acc_va vs. n_layers (prop. #grad_comps)']
  for (ax, title) in zip(axs, titles): 
    ax.legend(loc='lower right')
    ax.grid(); ax.set_title(title)
  # axs.legend(loc='upper left'); axs.grid();

  plt.show()


if __name__ == '__main__': main()



































