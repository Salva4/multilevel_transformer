import os
import torch
import sys

sys.path.append(os.path.join('..', '..', '..', 'src'))

from mgrit.mgrit import MGRIT

@torch.no_grad()
def solve_MGRIT(*args, **kwargs): 
  print('Using MGRIT')
  return MGRIT(*args, **kwargs)
