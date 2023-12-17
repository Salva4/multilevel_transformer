import os
import torch
import sys

sys.path.append(os.path.join('..', '..', '..', 'src'))

from mgrit.mgrit import MGRIT

@torch.no_grad()
def solve_mgrit(
  x0, mgrit_relaxation, mgrit_num_iterations, **other_fwd_details
):
  return MGRIT(
    x0, relaxation=mgrit_relaxation, num_iterations=mgrit_num_iterations,
    **other_fwd_details,
  )
