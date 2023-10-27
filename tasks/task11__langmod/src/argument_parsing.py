import argparse

def parse_arguments():
  parser = argparse.ArgumentParser()

  ## Data & training
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--lr', type=str, default='3e-4', help='lrlvl0_lrlvl1_...')
  parser.add_argument('--context_window', type=int, default=256)
  parser.add_argument('--input_text', type=str, default='shakespeare')
  parser.add_argument('--num_epochs', type=str, default='5000', help='10_10_10_10_10_...')
  parser.add_argument('--tokenization', type=str, default='gpt2', help='character|gpt2')

  ## Model
  parser.add_argument('--generate', action='store_true')
  parser.add_argument('--model_dimension', type=int, default=384)
  parser.add_argument('--model_name', type=str, default='transformer') # Linear, Transformer
  parser.add_argument('--N', type=int, default=6)
  parser.add_argument('--num_heads', type=int, default=6)

  ## Continuous model
  parser.add_argument('--coarsening_factor', type=int, default=2)
  parser.add_argument('--continuous', action='store_true')
  parser.add_argument('--interpol', type=str, default='constant')  # <-- always 'linear' in MG/OPT: I, R
  parser.add_argument('--levels_scheme', type=str, default='0', help='2_1_2_1_0_...')
  parser.add_argument('--mgopt', action='store_true')
  parser.add_argument('--mgrit', action='store_true')
  parser.add_argument('--num_iterations', type=int, default=2)
  parser.add_argument('--relaxation', type=str, default='F')
  parser.add_argument('--solver', type=str, default='Forward Euler')
  parser.add_argument('--T', type=float, default=None)

  ## Debugging, seed and saving
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--load', action='store_true')
  # parser.add_argument('--models_dir', type=str, default=None)
  # parser.add_argument('--output_fn', type=str, default=None)
  parser.add_argument('--save', action='store_true')
  parser.add_argument('--seed', type=int, default=0)#1337)

  args = parser.parse_args()
  return args

# parser.add_argument('--init', type=str, required=True)#default='xavier')
# parser.add_argument('--pe', type=str, required=True)#default='torch')


