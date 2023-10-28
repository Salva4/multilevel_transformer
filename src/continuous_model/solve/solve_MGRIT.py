import os
import torch
import sys

sys.path.append(os.path.join('..', '..', '..', 'src'))

from mgrit.mgrit import MGRIT

@torch.no_grad()
def solve_MGRIT(
  x, mgrit_relaxation, mgrit_num_iterations, **other_fwd_details
): 
  return MGRIT(
    x, relaxation=mgrit_relaxation, num_iterations=mgrit_num_iterations, 
    **other_fwd_details,
  )
