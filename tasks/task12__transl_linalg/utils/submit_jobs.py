import os
import pyautogui as pag
import time

def main():
    time.sleep(5)

    for fn in sorted(os.listdir('../experiments')):
        if fn.startswith('exp'):
            path = f'../experiments/{fn}'
            pag.write(f'sbatch {path}')
            time.sleep(.1)
            pag.press('return')
            time.sleep(.1)

main()
