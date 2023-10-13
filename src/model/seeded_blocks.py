# import torch
# import torch.nn as nn

# class SeedednnModule(nn.Module):
#   def __init__(self, seed=None, **kwargs):
#     super().__init__()
    
#     if seed is not None:
#       assert type(seed) == int, 'seed must be an integer'
#       torch.manual_seed(seed)