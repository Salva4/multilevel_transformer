import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

## Colors palette
import seaborn as sns
import colorcet as cc

parser = argparse.ArgumentParser()
parser.add_argument('--fn', type=str, default=None)
args = parser.parse_args()

def main():
  ## Obtain file name
  if args.fn is not None: fn = args.fn
  else:
    for dir_nm in os.listdir('../outputs/'):
      if dir_nm.startswith('cont'): break
    assert dir_nm.startswith('cont')
    fns = []
    for fn in os.listdir(f'../outputs/{dir_nm}/outputs/'):
      if fn.startswith('Cont'): fns.append(fn)
    # print('Select the file:')
    # for (i, fn) in enumerate(fns):
    #   print(f'\t{i} - {fn}')
    # fn = fns[int(input('--> '))]
    # assert fn.startswith('Cont')

  fig, axs = plt.subplots(1,2)
  fig.suptitle(f'{dir_nm}')
  colors = sns.color_palette(cc.glasbey, len(fns))

  for j, fn in enumerate(fns):
    ## Extract losses and accs
    losses_tr, losses_va, accs_tr, accs_va = [], [], [], []
    with open(f'../outputs/{dir_nm}/outputs/{fn}', 'r') as f:
      for line in f:
        if line.startswith('training loss'):
          loss, acc = line.strip().split('\t')
          loss, acc = float(loss.split()[-1]), float(acc.split()[-1][:-1])
          losses_tr.append(loss)
          accs_tr.append(acc)

        if line.startswith('validation loss'):
          loss, acc = line.strip().split('\t')
          loss, acc = float(loss.split()[-1]), float(acc.split()[-1][:-1])
          losses_va.append(loss)
          accs_va.append(acc)

    ## Filter name
    replace = 'ContTrans_- nnodes-#n ntasksxnode-#tn proc- .txt-'
    for s in replace.split(): fn = fn.replace(*s.split('-'))
    fn = fn.replace('_', ' ').strip()

    ## Plot
    # axs[0].plot(losses_tr, 'b')
    axs[0].plot(losses_va, color=colors[j], label=f'{fn}')
    # axs[1].plot(accs_tr, 'b', label='acc tr')
    axs[1].plot(accs_va, color=colors[j], label=f'{fn}')

  axs[0].legend(); axs[1].legend()
  axs[0].grid(); axs[1].grid()
  plt.show()

if __name__ == '__main__': main()



































