
def obtain_model_name(variables_dictionary):
  model_name = ''
  for (k, v) in sorted(variables_dictionary.items()):
    if v is None: continue
    if k == 'batch_size'       : k = 'bs'
    if k == 'coarsening_factor': k = 'cf'
    if k == 'context_window'   : k = 'L'
    if k == 'continuous'       : k = 'cont'
    if v == False              : v = 'F'
    if v == True               : v = 'T'
    if k == 'input_text'       : k = 'text'
    if v == 'shakespeare'      : v = 'shak'
    if v == 'wikipedia'        : v = 'wiki'
    if k == 'levels_scheme'    : k = 'sch'
    if k == 'save'             : continue
    if k == 'model_dimension'  : k = 'd'
    if k == 'model_name'       : k = ''
    if k == 'num_epochs'       : k = 'ep'
    if k == 'num_heads'        : k = 'H'
    if v == 'Forward Euler'    : v = 'FE'
    if k == 'tokenization'     : k = 'tok'
    if v == 'character'        : v = 'char'
    if k == 'load'             : continue
    if k == 'num_training_batches'      : k = 'ntb'
    if k == 'num_validation_batches'    : k = 'nvb'
    if k == 'gradient_accumulation_size': k = 'gas'
    if k == 'gradient_clipping_norm'    : k = 'gcn'
    if k == 'optimizer_name'            : k = 'opt'
    if k == 'learning_rate'             : k = 'lr'
    if k == 'momentum'                  : k = 'mom'
    if k == 'dim_ff'                    : k = 'ff'
    if k == 'dropout'                   : k = 'do'
    if k == 'num_encoder_layers'        : k = 'Nenc'
    if k == 'num_decoder_layers'        : k = 'Ndec'
    if k == 'encoder_T'                 : k = 'Tenc'
    if k == 'decoder_T'                 : k = 'Tdec'
    if k == 'ode_solver'                : k = ''
    if k == 'multilevel_interpolation'  : k = 'inp'
    if not variables_dictionary['use_mgrit'] and k in [
    'use_mgrit', 'mgrit_relaxation', 'mgrit_num_iterations'
    ]: continue
    else: pass
    if not variables_dictionary['use_mgopt'] and k in [
      'use_mgopt', 'mgopt_mu', 'mgopt_mu_coarsest', 'mgopt_nu', 
      'mgopt_num_levels', 'mgopt_cycle', 'mgopt_num_iterations',
    ]: continue
    else:
      if k == 'use_mgopt'           : k = 'mgopt'
      if k == 'mgopt_mu'            : k = 'mu'
      if k == 'mgopt_mu_coarsest'   : k = 'muc'
      if k == 'mgopt_nu'            : k = 'nu'
      if k == 'mgopt_num_levels'    : k = 'mtlv'
      if k == 'mgopt_cycle'         : k = 'cy'
      if k == 'mgopt_num_iterations': k = 'mtit'


    model_name += f'_{k}{v}'

  model_name = model_name[1:]
  model_name1 = model_name + '_1'
  model_name2 = model_name + '_2'
  return model_name1, model_name2




