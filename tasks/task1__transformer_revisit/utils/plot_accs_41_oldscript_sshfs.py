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

sync_dir_1 = '/Users/marcsalvado/Desktop/SCRIPTS/4-MLT/sync_torchbraid_MLPQ/'
sync_dir_2 = 'ML_PQ'#'torchbraid/examples/transformer'
if os.listdir(sync_dir_1) == []:
    print('Error: cal fer sshfs')
    exit()
os.chdir(sync_dir_1 + sync_dir_2)

# other_list = '''
# ContTrans_lr1e-2_initNormal_peTorch_T01_N5-9-17-33_numepochs1280-1280-1280-1280_interpollinear_batch8_lrfactor1.0.txt
# '''.strip().split()
other_list = '''
'''.strip().split()

parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='transformer')
parser.add_argument('--order', type=str, default='horizontally')
parser.add_argument('--rc', type=str, default=None)
parser.add_argument('--hspace', type=float, default=.5)
parser.add_argument('--wspace', type=float, default=.2)
parser.add_argument('--input', type=int, default=None)
parser.add_argument('--highlight', type=str, default=None)
args = parser.parse_args()

assert args.order in ['horizontally', 'vertically']

def commondiff_titles(l):
    def common_(l_sects, i):
        for n in range(len(l_sections) - 1):
            if l_sects[n][i] != l_sects[n+1][i]:
                return False

        return True

    l_sections = []
    for filename in l:
        sections = filename.split('_')
        l_sections.append(sections)

    i_common = []
    i_diff = []
    n = len(l_sections[0])
    for i in range(n):
        if common_(l_sections, i):
            i_common.append(i)
        else:
            i_diff.append(i)

    common = ' '.join([l_sections[0][i] for i in i_common])

    return common, i_diff
        

def obtain_some_rows_cols(n):
    rows = cols = math.ceil(np.sqrt(n))
    if rows*(cols - 1) >= n:
        cols -= 1
    # print(f'rows: {rows}, cols {cols}')
    return rows, cols

def focus_tens(valaccs):
    maxnum = max(valaccs)
    maxten = int(10*np.ceil(maxnum/10))
    minten = max(0, maxten - 50)
    return minten, maxten

def get_model(inp):
    assert inp == None or type(inp) == int
    # listdir = sorted(os.listdir('outputs'), key=lambda z: os.path.getctime(f'outputs/{z}'))
    listdir = sorted(os.listdir('outputs'))
    listdir.remove('old_models')
    if inp == None:
        if len(listdir) > 1:
            print('Select model index:')
            for i, x in enumerate(listdir):
                print(f'\t{i} - {x}')
            model_idx = int(input('Write index: '))
        else:
            model_idx = 0
            print(f'Selecting {listdir[0]}')
    else:
        model_idx = inp
    return listdir[model_idx]

def timestr_(s):
    s = int(s)
    h, m, s = s//3600, (s//60)%60, s%60
    t = (f'{h}h ' if h > 0 else '') \
        + (f'{str(m).zfill(2)}min ' if (m > 0 or h > 0) else '') \
        + f'{str(s).zfill(2)}s'
    return t

def main():
    assert 'outputs' in os.listdir()

    # model = args.model.lower()
    model = get_model(args.input)
    file_path = f'outputs/{model}/outputs/'
    list_dir = os.listdir(file_path)
    r = re.compile('.*txt')
    list_outs = list(filter(r.match, list_dir))

    # fother = lambda filename: not('epochs1_' in filename)
    # other_list = sorted(list(filter(fother, list_outs)))

    list_to_run = sorted(list_outs) if other_list == [] else other_list

    list_to_run = list(filter(lambda z: z[:len('error')] != 'error', list_to_run))

    common, i_diff = commondiff_titles(list_to_run)
    simplify = 'usedowncycle-down True-T False-F epochs-ep cfactor-cf levels-lvs nnodes-#n ntasksxnode-#txn proc- batchsize-bchsz maxiters-mxit N-n T-t'.strip().split()
    for change in simplify:
        common = common.replace(*change.split('-'))

    rows, cols = obtain_some_rows_cols(len(list_to_run)) if args.rc == None \
      else (int(args.rc.split('-')[0]), int(args.rc.split('-')[1]))
    fig, axs = plt.subplots(rows, cols)
    fig.suptitle(f'Accuracy (tr:blue // val:red) - {model}\n{common}', fontsize=15)

    times = []
    valaccs_max_file = []

    for j, file_name in enumerate(list_to_run):
        file = file_path + file_name
        traccs, valaccs = [], []
        valacc_max = -1.
        valaccs_max_model = []
        nextt = False
        with open(file, 'r') as f:
            epoch_ctr, changes, time = 0, [], 0.
            for i in f:
                i = i.strip()
                text = '(Batch-)'
                if i[:len(text)] == text:
                    print(i)
                    r = re.compile('\d*\.\d*%')
                    tracc, valacc = r.findall(i)
                    tracc, valacc = float(tracc[:-1]), float(valacc[:-1])
                    valaccs.append(valacc)
                    epoch_ctr += 1
                    valacc_max = max(valacc_max, valacc)
                elif i == '------------------- ------------------- ------------------- ------------------- ---------- ----------':
                    nextt = True
                elif nextt == True: 
                    nextt = False
                    time_str = ' '.join(i.split()[:-1]).strip().split()[-1]
                    if time_str[:2] == '1-':
                        time = 24*3600
                        time_str = time_str[2:]
                    time += sum([60**i*int(time_str.split(':')[-(i+1)]) for i in range(3)])
        print(epoch_ctr)
        if valacc_max != -1:
            valaccs_max_model.append(valacc_max)
        valaccs_max_file.append(valaccs_max_model)

        print(valaccs)

        x, y = (j//cols, j%cols) \
          if args.order == 'horizontally' else (j%rows, j//rows)
        try:
            ax = axs[x, y]
        except:
            try:
                ax = axs[j]
            except:
                ax = axs

        sections = file_name[:-4].split('_')
        diff = [sections[i] for i in i_diff]
        title = ' '.join(diff)
        simplify = 'usedowncycle-down True-T False-F epochs-ep cfactor-cf levels-lvs nnodes-#n ntasksxnode-#txn proc- batchsize-bchsz maxiters-mxit N-n T-t'.strip().split()
        for change in simplify:
            title = title.replace(*change.split('-'))
        simplify = '#n- #txn- gpu- initxavier- petorch- ep10000- btchsz16- nlvls2- nVcycles1-'.strip().split()
        for change in simplify:
            title = title.replace(*change.split('-'))

        # title = title[:len(title)//2] + '\n' + title[len(title)//2:]
        
        print(args.highlight, title)
        if args.highlight != None:
            title = title.replace(args.highlight, args.highlight.upper())

        ax.set_title(title)

        if valaccs == []:
            continue

        ax.plot(range(0, len(traccs)), traccs, 'b', label='Training accuracy')
        ax.plot(range(0, len(valaccs)), valaccs, 'r', label='Validation accuracy') 
        ax.vlines(changes, 0, 100, color='black', linestyle='--')
        # ax.legend()
        ax.grid(True)
        minten, maxten = focus_tens(valaccs)
        # minten, maxten = 0, 100
        ax.set_ylim(minten, maxten)
        delta = 10
        ax.set_yticks(range(minten, maxten + delta, delta))
        seconds, minutes, hours = time%60, int((time//60)%60), int(time//3600)
        ax.text(1, maxten-10, f'Running time: {hours} h, {minutes} m, {seconds : .2f} s')

        times.append(('\n'.join(np.array(title.split())), time))

    for k in range(j+1, rows*cols):
        x, y = (k//cols, k%cols) \
          if args.order == 'horizontally' else (k%rows, k//rows)
        try:
            ax = axs[x, y]
        except:
            try:
                ax = axs[k]
            except:
                ax = axs

        ax.set_facecolor('gray')


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
    # figure, ax = plt.subplots()
    # ax.set_axis_off()
    # row_headers = [i for i in list_to_run]
    # column_headers = ['max val acc', 'running time (s)']
    # cell_accs = [str(i) for i in valaccs_max_file]
    # cell_times = [timestr_(i[1]) for i in times]
    # cell_text = np.array([cell_accs, cell_times], dtype=str).T
    # rcolors = plt.cm.RdBu(np.full(len(row_headers), 0.1))
    # ccolors = plt.cm.RdBu(np.full(len(column_headers), 0.9))
    # rcolors = np.full(len(row_headers), 'darkturquoise')
    # ccolors = ['gold', 'firebrick']
    # the_table = ax.table(
    #     cellText=cell_text,
    #     rowLabels=row_headers,
    #     rowColours=rcolors,
    #     rowLoc='right',
    #     colColours=ccolors,
    #     colLabels=column_headers,
    #     loc='center',
    #     bbox=[0, 0, 1, 1],
    # )
    # plt.xlim(-30, 30)
    # plt.show()
    print('FILE\tMAX VAL ACC\tRUNNING TIME (s)')
    for i, x in enumerate(list_to_run):
        try:
            print(f'{x}\n\tMAX VAL ACC:\t{valaccs_max_file[i]}\t\tRUNNING TIME (s):\t{timestr_(times[i][1])})')
        except:
            pass


if __name__=='__main__':
    ## Removes files: .DS_Store, __pycache__/*
    # remove_undesired.do()
    main()
    # remove_undesired.do()








































