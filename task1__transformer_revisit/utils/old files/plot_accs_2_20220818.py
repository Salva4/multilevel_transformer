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
ContTrans_lr1e-2_initNormal_peTorch_T05_N33_numepochs5120_interpolconstant_batch8_lrfactor1.0.txt
ContTrans_lr1e-2_initNormal_peTorch_T05_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor0.5.txt
'''.strip().split()
#ContTrans_lr1e-2_initNormal_peTorch_T05_N5-9-17-33_numepochs1280-640-320-160_interpollinear_batch8_lrfactor1.0.txt

## N33 has a lr discount factor of .5 each 1280 epochs (4 in total)

colors = ['#003147', '#D12959', '#009ADE', 'dimgray', '#00718B', '#F2ACCA', 'gold', '#FED976']
labels = [
    '1 level',#, N=33, 5120 batch passes',
    '4 levels',#, N=5-9-17-33, 5120 batch passes',
    '4 levels',#, N=5-9-17-33, 2400 batch passes',
]
linestyles = ['solid', 'dotted', 'dashed']
gradevals = [
    [i for i in range(0, 1280*4, 10)],
    [i/8 for i in range(0, 1280, 10)] + [1280/8 + i/4 for i in range(0, 1280, 10)] \
     + [1280/8+1280/4 + i/2 for i in range(0, 1280, 10)] + [1280/8+1280/4+1280/2 + i for i in range(0, 1280, 10)],
    [i/8 for i in range(0, 1280, 10)] + [1280/8 + i/4 for i in range(0, 640, 10)] \
     + [1280/8+640/4 + i/2 for i in range(0, 320, 10)] + [1280/8+640/4+320/2 + i for i in range(0, 160, 10)],
]

parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='transformer')
parser.add_argument('--order', type=str, default='horizontally')
parser.add_argument('--rc', type=str, default=None)
parser.add_argument('--hspace', type=float, default=.5)
parser.add_argument('--wspace', type=float, default=.2)
args = parser.parse_args()

assert args.order in ['horizontally', 'vertically']

def focus_tens(traccs, valaccs):
    maxnum = max(max(traccs), max(valaccs))
    maxten = int(10*np.ceil(maxnum/10))
    minten = max(0, maxten - 50)
    return minten, maxten

def timestr_(s):
    s = int(s)
    h, m, s = s//3600, (s//60)%60, s%60
    t = (f'{h}h ' if h > 0 else '') \
        + (f'{str(m).zfill(2)}min ' if (m > 0 or h > 0) else '') \
        + f'{str(s).zfill(2)}s'
    return t

def main():
    assert os.getcwd()[-5:] == 'ML_PQ' and 'outputs' in os.listdir()

    model = 'continuous_transformer__20220817_03_interpolcomparison'
    file_path = f'outputs/{model}/'
    fig, ax = plt.subplots()

    times = []
    valaccs_max_file = []
    list_to_run = other_list
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

    ## Uncomment for plot 1 #################
        ax.plot(range(0, 10*len(valaccs), 10), valaccs, color=colors[j],#'r', 
            label=labels[j], linewidth=2, linestyle=linestyles[j]) 
        # valaccs = np.array(valaccs)/100
        # ax.semilogy(range(0, 10*len(valaccs), 10), valaccs, color=colors[j],#'r', 
        #     label=labels[j]) 

        # ax.vlines(changes, 0, 100, color='black', linestyle='--')
        # ax.legend()
        try:
            minten, maxten = focus_tens(traccs, valaccs)
            minten, maxten = 40, 90
            ax.set_ylim(minten, maxten)
            delta = 5
            ax.set_yticks(range(minten, maxten + delta, delta))
        except:
            pass

    fig.suptitle(f'Continuous transformer for morphology classification', fontsize=25)
    ax.grid(True)
    ax.set_xlabel('epochs', fontsize=25)
    ax.set_ylabel('validation accuracy (%)\n', fontsize=25)#(%)', fontsize=15)
    ax.set_xlim(0, 5000)
    ax.legend(loc='lower right', fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    #########################################

    ## Uncomment for plot 2 #################
    #     gradeval = gradevals[j]
    #     ax.plot(gradeval, valaccs, color=colors[j], 
    #         label=labels[j], linewidth=2, linestyle=linestyles[j])

    # minten, maxten = 40, 90
    # ax.set_ylim(minten, maxten)
    # delta = 5
    # ax.set_yticks(range(minten, maxten + delta, delta))
    
    # fig.suptitle(f'Continuous transformer for morphology classification', fontsize=25)
    # ax.grid(True)
    # ax.set_xlabel('effective gradient evaluations',#'gradient evaluation (divided by #parameters of finest model)', 
    #     fontsize=25)
    # ax.set_ylabel('validation accuracy (%)\n', fontsize=25)#(%)', fontsize=15)
    # ax.set_xlim(0, 5000)
    # ax.legend(loc='lower right', fontsize=15)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    
    # print(gradevals)

    # plt.show()
    #########################################


if __name__=='__main__':
    ## Removes files: .DS_Store, __pycache__/*
    remove_undesired.do()
    main()
    remove_undesired.do()








































