import argparse

def parse_arguments():
  parser = argparse.ArgumentParser()

  ## Debugging, seed and saving
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--models_dir', type=str, default=None)
  parser.add_argument('--output_fn', type=str, default=None)
  # parser.add_argument('--seed', type=int, default=0)

  ## Data & training
  parser.add_argument('--batch_size', type=int, default=8)#64
  parser.add_argument('--lr', type=str, default='1e-2', help='lrlvl0_lrlvl1_...')
  parser.add_argument('--max_len', type=int, default=2048)
  parser.add_argument('--momentum', type=str, default='.9', help='momentumlvl0_momentumlvl1_...')
  parser.add_argument('--num_epochs', type=str, default='1000000000', help='10_10_10_10_10_...')
  parser.add_argument('--optimizer', type=str, default='SGD')

  ## Model
  parser.add_argument('--model_dimension', type=int, default=128)
  parser.add_argument('--model_name', type=str, default='transformer') # Linear, Transformer
  parser.add_argument('--num_heads', type=int, default=1)
  parser.add_argument('--N', type=int, default=4)

  ## Continuous model
  parser.add_argument('--coarsening_factor', type=int, default=2)
  parser.add_argument('--continuous', action='store_true')
  parser.add_argument('--interpol', type=str, default='constant')  # <-- always 'linear' in MG/OPT: I, R
  parser.add_argument('--levels_scheme', type=str, default='0', help='2_1_2_1_0_...')
  parser.add_argument('--solver', type=str, default='Forward Euler')
  parser.add_argument('--T', type=float, default=None)

  args = parser.parse_args()
  return args

# parser.add_argument('--init', type=str, required=True)#default='xavier')
# parser.add_argument('--pe', type=str, required=True)#default='torch')
