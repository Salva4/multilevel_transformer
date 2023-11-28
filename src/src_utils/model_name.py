
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
    if k == 'levels_scheme'    : k = 'scheme'
    if k == 'save'             : continue
    if k == 'model_dimension'  : k = 'd'
    if k == 'model_name'       : k = ''
    if k == 'num_epochs'       : k = 'epochs'
    if k == 'num_heads'        : k = 'H'
    if v == 'Forward Euler'    : v = 'FE'
    if k == 'tokenization'     : k = 'tok'
    if v == 'character'        : v = 'char'
    if k == 'load'             : continue

    model_name += f'_{k}{v}'

  model_name = model_name[1:]
  model_name1 = model_name + '_copy1'
  model_name2 = model_name + '_copy2'
  return model_name1, model_name2




