import argparse

def parse_arguments():
  parser = argparse.ArgumentParser()

  ## Data & training
  parser.add_argument('--batch_size'            , type=int, default=4)#64)
  parser.add_argument('--num_training_batches'  , type=int, default=None)#100)
  parser.add_argument('--num_validation_batches', type=int, default=None)#100)
  parser.add_argument('--num_epochs', type=str, default='2', help='10_10_10_10_10_...')

  ## Optimizer
  parser.add_argument('--optimizer_name', type=str, default='Adam')
  parser.add_argument('--learning_rate' , type=str, default=None, help='lrlvl0_lrlvl1_...')  
  parser.add_argument('--momentum'      , type=str, default=None, help='momentumlvl0_momentumlvl1_...')

  ## Model
  parser.add_argument('--model_name'        , type=str, default='transformer') # Linear, Transformer
  parser.add_argument('--model_dimension'   , type=int, default=512          )
  parser.add_argument('--num_heads'         , type=int, default=8            )
  parser.add_argument('--dim_ff'            , type=int, default=2048         )
  parser.add_argument('--num_encoder_layers', type=int, default=6            )
  parser.add_argument('--num_decoder_layers', type=int, default=6            )

  ## Continuous model
  parser.add_argument('--continuous', action='store_true'     )
  parser.add_argument('--encoder_T' , type=float, default=None)
  parser.add_argument('--decoder_T' , type=float, default=None)
  parser.add_argument('--ode_solver', type=str  , default=None)

  ## Multilevel
  parser.add_argument('--levels_scheme'           , type=str, default=None, help='2_1_2_1_0_...')
  parser.add_argument('--coarsening_factor'       , type=int, default=None,                     )
  parser.add_argument('--multilevel_interpolation', type=str, default=None,                     )  # <-- always 'linear' in MG/OPT: I, R

  ## MGRIT
  parser.add_argument('--use_mgrit'           , action='store_true'   )
  parser.add_argument('--mgrit_relaxation'    , type=str, default=None)
  parser.add_argument('--mgrit_num_iterations', type=int, default=None)

  ## MGOPT
  parser.add_argument('--use_mgopt'           , action='store_true'   )
  parser.add_argument('--mgopt_mu'            , type=int, default=None)
  parser.add_argument('--mgopt_mu_coarsest'   , type=int, default=None)
  parser.add_argument('--mgopt_nu'            , type=int, default=None)
  parser.add_argument('--mgopt_num_levels'    , type=int, default=None)
  parser.add_argument('--mgopt_cycle'         , type=str, default=None)
  parser.add_argument('--mgopt_num_iterations', type=int, default=None)

  ## Debugging, seed and saving
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--seed' , type=int, default=0)
  parser.add_argument('--save' , action='store_true')
  parser.add_argument('--load' , action='store_true')
  # parser.add_argument('--models_dir', type=str, default=None)
  # parser.add_argument('--output_fn', type=str, default=None)

  args = parser.parse_args()
  return args

def assert_and_correct_arguments(args):
  ## False --> False/None
  false_implies_falsenone = {
    'continuous': [
      'encoder_T', 'decoder_T', 'ode_solver',
      'levels_scheme', 'coarsening_factor', 'multilevel_interpolation',
      'use_mgrit', 'use_mgopt',
    ],
    'use_mgrit': ['mgrit_relaxation', 'mgrit_num_iterations'],
    'use_mgopt': [
      'mgopt_mu', 'mgopt_nu', 'mgopt_num_levels', 'mgopt_cycle',
      'mgopt_num_iterations',
    ],
  }

  for (k, v_list) in false_implies_falsenone.items():
    if not args.__dict__[k]:
      for v in v_list:
        assert args.__dict__[v] in [None, False]

  ## True --> False
  # true_implies_false = {
  #   'use_mgrit': ['use_mgopt'],
  #   'use_mgopt': ['use_mgrit'],
  # }
  #
  # for (k, v) in true_implies_false.items():
  #   if args.__dict__[k]:
  #     for v in v_list:
  #       assert args.__dict__[v] == False

  ## Default values
  num_level_changes = len(args.levels_scheme.split('_')) \
                      if args.levels_scheme is not None else 1
  default_values = {
    'learning_rate': '_'.join(['1e-4']*num_level_changes),
    'momentum'     : '_'.join(['0.'  ]*num_level_changes) \
                     if args.optimizer_name == 'SGD' else None,
    'encoder_T': float(args.num_encoder_layers),
    'decoder_T': float(args.num_decoder_layers),
    'ode_solver': 'Forward Euler',
    'levels_scheme': '0',
    'coarsening_factor': 2,
    'multilevel_interpolation': 'constant',
    'mgrit_relaxation': 'F',
    'mgrit_num_iterations': 2,
    'mgopt_mu': 1,
    'mgopt_mu_coarsest': 1,
    'mgopt_nu': 1,
    'mgopt_num_levels': 2,
    'mgopt_cycle': 'V',
    'mgopt_num_iterations': 1,
  }

  for (k, v) in default_values.items():
    if args.__dict__[k] is None: args.__dict__[k] = v




