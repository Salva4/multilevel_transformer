
def obtain_model_name(_vars):
  model_name = ''
  for (k, v) in sorted(_vars.__dict__.items()):
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

def load_model(model, optimizer, model_name1, model_name2):
  other_states = {}

  try:
    print('Loading model, copy1')
    other_states = model.load(model_name=model_name1, optimizer=optimizer)
    print('other_states', other_states)
  except:
    try:
      print('Loading model, copy2')
      other_states = model.load(model_name=model_name2, optimizer=optimizer)
    except:
      # print('The model could not be loaded because of an unknown error.')
      other_states['error'] = 'Unknown error.'
  
  if 'error' in other_states: print(f"Error: {other_states['error']}")
  else                      : print('Model successfully loaded.')

def generate_text(model, device, decoding_function, max_new_tokens, **kwargs):
  model.eval()
  bos_token = '<|endoftext|>'
  bos_token_id = 50256#tokenizer('<|endoftext|>')['input_ids'][0]
  context = torch.empty((1, 1), dtype=torch.long, device=device).fill_(bos_token_id)
  print(
    decoding_function(
      generate(
        model=model, x=context, max_new_tokens=max_new_tokens, **kwargs,
      )[0].tolist()
    )
  )
  #open('../data/more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
