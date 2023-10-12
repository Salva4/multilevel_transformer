import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as prod
import argparse
import re
import math
import datetime as dt

sys.path.append('utils')
import remove_undesired

other_list = '''
ContTrans_lr1e-2_initNormal_peTorch_T01_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T01_N33_numepochs5120_interpolconstant_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T01_N5-9-17-33_numepochs1280-640-320-160_interpollinear_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T01_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor0.5.txt
ContTrans_lr1e-2_initNormal_peTorch_T01_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor0.8.txt
ContTrans_lr1e-2_initNormal_peTorch_T05_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T05_N33_numepochs5120_interpolconstant_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T05_N5-9-17-33_numepochs1280-640-320-160_interpollinear_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T05_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor0.5.txt
ContTrans_lr1e-2_initNormal_peTorch_T05_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor0.8.txt
ContTrans_lr1e-2_initNormal_peTorch_T10_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T10_N33_numepochs5120_interpolconstant_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T10_N5-9-17-33_numepochs1280-640-320-160_interpollinear_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T10_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor0.5.txt
ContTrans_lr1e-2_initNormal_peTorch_T10_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor0.8.txt
'''.strip().split()

parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='transformer')
parser.add_argument('--order', type=str, default='horizontally')
parser.add_argument('--rc', type=str, default=None)
parser.add_argument('--hspace', type=float, default=.5)
parser.add_argument('--wspace', type=float, default=.2)
args = parser.parse_args()

assert args.order in ['horizontally', 'vertically']

def obtain_some_rows_cols(n):
    rows = cols = math.ceil(np.sqrt(n))
    if rows*(cols - 1) >= n:
        cols -= 1
    # print(f'rows: {rows}, cols {cols}')
    return rows, cols

def focus_tens(traccs, valaccs):
    maxnum = max(max(traccs), max(valaccs))
    maxten = int(10*np.ceil(maxnum/10))
    minten = max(0, maxten - 50)
    return minten, maxten

def get_model():
    print('Select model index:')
    listdir = sorted(os.listdir('outputs'), key=lambda z: os.path.getctime(f'outputs/{z}'))
    for i, x in enumerate(listdir):
        print(f'\t{i} - {x}')
    model_idx = int(input('Write index: '))
    return listdir[model_idx]

def timestr_(s):
    s = int(s)
    h, m, s = s//3600, (s//60)%60, s%60
    t = (f'{h}h ' if h > 0 else '') \
        + (f'{str(m).zfill(2)}min ' if (m > 0 or h > 0) else '') \
        + f'{str(s).zfill(2)}s'
    return t

def main():
    assert os.getcwd()[-5:] == 'ML_PQ' and 'outputs' in os.listdir()

    # model = args.model.lower()
    model = get_model()
    file_path = f'outputs/{model}/'
    list_dir = os.listdir(file_path)
    r = re.compile('.*txt')
    list_outs = list(filter(r.match, list_dir))
    rows, cols = obtain_some_rows_cols(len(list_outs)) if args.rc == None \
      else (int(args.rc.split('-')[0]), int(args.rc.split('-')[1]))
    fig, axs = plt.subplots(rows, cols)
    fig.suptitle(f'Accuracy (tr:blue // val:red) - {model}\n(lr1e-2 initNormal peTorch batch8 int.linear)', fontsize=15)

    times = []
    valaccs_max_file = []
    list_to_run = sorted(list_outs) if other_list == [] else other_list
    for j, file_name in enumerate(list_to_run):
        file = file_path + file_name
        traccs, valaccs = [], []
        valacc_max = -1.
        valaccs_max_model = []
        with open(file, 'r') as f:
            epoch_ctr, changes, time = 0, [], 0.
            for i in f:
                i = i.strip()
                if 'Tr/Va' in i:
                    l = i.split('\t')
                    tracc, valacc = float(l[2].strip()[:-1]), float(l[3].strip()[:-1]) #[1], [2]
                    traccs.append(tracc)
                    valaccs.append(valacc)
                    epoch_ctr += 1
                    valacc_max = max(valacc_max, valacc)
                elif 'Training finished' in i:
                    changes.append(epoch_ctr*10)
                    valaccs_max_model.append(valacc_max)
                    valacc_max = -1.
                elif 'Time:' in i:
                    l = i.split()
                    time = float(l[3])

        if valacc_max != -1:
            valaccs_max_model.append(valacc_max)
        valaccs_max_file.append(valaccs_max_model)

        x, y = (j//cols, j%cols) \
          if args.order == 'horizontally' else (j%rows, j//rows)
        ax = axs[x, y]
        ax.plot(range(0, 10*len(traccs), 10), traccs, 'b', label='Training accuracy')
        ax.plot(range(0, 10*len(valaccs), 10), valaccs, 'r', label='Validation accuracy') 
        ax.vlines(changes, 0, 100, color='black', linestyle='--')
        # ax.legend()
        ax.grid(True)
        try:
            minten, maxten = focus_tens(traccs, valaccs)
            minten, maxten = 0, 100
            ax.set_ylim(minten, maxten)
            delta = 10
            ax.set_yticks(range(minten, maxten + delta, delta))
        except:
            pass
        title = ' '.join(file_name[:-4].split('_')[1:])
        title = ' '.join(title.split()[3:]) if 1 or j != 0 else title
        title = ' '.join(title.split()[:4] + title.split()[5:]) if 1 or j != 0 else title
        title = title.replace('numepochs', '#epochs')
        title = title.replace('interpol', 'int.')
        title = title.replace(' lrfactor1.0', '')
        title = title.replace(' int.linear', '')
        title = title.replace(' int.constant', '')
        ax.set_title(title)
        seconds, minutes, hours = time%60, int((time//60)%60), int(time//3600)
        ax.text(1, maxten-10, f'Running time: {hours} h, {minutes} m, {seconds : .2f} s')

        times.append(('\n'.join(np.array(title.split())), time))


    plt.subplots_adjust(hspace=args.hspace, wspace=args.wspace)
    # plt.savefig(f'acc_{model}')
    plt.show()

    # ## Running time 
    # plt.figure()
    # plt.bar([i[0] for i in times], [i[1] for i in times], 
    #         color=['darkblue']*6+['firebrick']*6+['darkgreen']*6)
    # plt.xlabel('experiment')
    # plt.ylabel('computing time (s)')
    # plt.legend()
    # plt.title('Val accuracy\nblue: 1 level // red: 3 levels')
    # plt.show()

    # ## Max acc
    # plt.figure()
    # bar = np.array([(i, y, j) for (i, x) in enumerate(valaccs_max_file) for (j, y) in enumerate(x)])
    # xmod = 1 + (bar[:, 2] - 1)*.25
    # # print(valaccs_max_file)
    # # print(bar)
    # plt.bar(bar[:, 0]+xmod, bar[:, 1], .2,
    #         color=np.array(['darkblue', 'firebrick', 'darkgreen'])[bar.astype(int)[:, 2]])
    # plt.grid(True)
    # plt.yticks(range(0, 105, 5))
    # plt.xticks(range(1, 19))
    # plt.title('Max val accuracy per grid (coarse, fine) and per experiment\ncoarsest:blue // medium:red // finest:green')
    # plt.show()

    ## Table
    figure, ax = plt.subplots()
    ax.set_axis_off()
    row_headers = [f'T={[1,5,10][i//5]} exp{i%5+1}'.replace('exp5', 'exp5.1').replace('exp6', 'exp5.2') \
                   for i in range(len(valaccs_max_file))]
    column_headers = ['max val acc', 'running time (s)']
    cell_accs = [str(i) for i in valaccs_max_file]
    cell_times = [timestr_(i[1]) for i in times]
    cell_text = np.array([cell_accs, cell_times], dtype=str).T
    # print(cell_text)
    # rcolors = plt.cm.RdBu(np.full(len(row_headers), 0.1))
    # ccolors = plt.cm.RdBu(np.full(len(column_headers), 0.9))
    rcolors = np.full(len(row_headers), 'darkturquoise')
    ccolors = ['gold', 'firebrick']
    the_table = ax.table(
        cellText=cell_text,
        rowLabels=row_headers,
        rowColours=rcolors,
        rowLoc='right',
        colColours=ccolors,
        colLabels=column_headers,
        loc='center'
    )
    plt.show()

if __name__=='__main__':
    ## Removes files: .DS_Store, __pycache__/*
    remove_undesired.do()
    main()
    remove_undesired.do()








































