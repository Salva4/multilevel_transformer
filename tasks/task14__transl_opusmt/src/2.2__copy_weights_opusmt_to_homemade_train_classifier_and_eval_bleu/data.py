import datasets
import os
from torch.utils.data import DataLoader
import tqdm

# lang_src, lang_tgt = 'de', 'en'
lang_src, lang_tgt = 'en', 'de'
dir_data = '../data/deen_translation/'
if 'data' not in os.listdir('..'): dir_data = '../' + dir_data
fn = {
  'train': {
    'src': 'train.y',#'train.x', 
    'tgt': 'train.x',#'train.y'
  }, 
  'test': {
    'src': 'interpolate.y',#'interpolate.x', 
    'tgt': 'interpolate.x',#'interpolate.y',
  },
}

def obtain_data(_vars):
  ds = {  # init
    'train': {'translation': []},
     'test': {'translation': []},
  }
  
  for mode in ['train', 'test']:  # extract sentences from files 
    path_src = open(os.path.join(dir_data, fn[mode]['src']), 'r')
    path_tgt = open(os.path.join(dir_data, fn[mode]['tgt']), 'r')

    print(f'Obtaining "{mode}" data')
    for i, (line_src, line_tgt) in enumerate(tqdm.tqdm(zip(path_src, path_tgt))):
      if _vars.debug and i > 100: break#20000: break
      ds[mode]['translation'].append(
        {
          lang_src: line_src.strip(),
          lang_tgt: line_tgt.strip(),
        }
      )

  ## dict --> Dataset --> DatasetDict + tokenize
  ds_tr = datasets.Dataset.from_dict(ds['train'])
  ds_te = datasets.Dataset.from_dict(ds['test'])
  ds_dict = datasets.DatasetDict({'train': ds_tr, 'test': ds_te})
  ds_dict = ds_dict.map(
    lambda partition: φ_preprocess(partition, _vars.tokenizer), 
    batched=True
  ).with_format('torch', device=_vars.dev)

  dl = {}
  dl['train'] = DataLoader(
    ds_dict['train'], 
    batch_size=_vars.batch_size, 
    shuffle=True
  )
  dl['test'] = DataLoader(
    ds_dict['test'], 
    batch_size=_vars.batch_size, 
    shuffle=False
  )

  _vars.lang_src, _vars.lang_tgt = lang_src, lang_tgt
  _vars.ds, _vars.dl = ds_dict, dl

## Adapted from https://huggingface.co/docs/transformers/tasks/translation
def φ_preprocess(partition, tokenizer):
  inputs  = [instance[lang_src] for instance in partition['translation']]
  targets = [instance[lang_tgt] for instance in partition['translation']]
  model_inputs = tokenizer(
    inputs, 
    text_target=targets, 
    max_length=128, 
    # truncation=True,
    padding='max_length'
  )
  return model_inputs
    




































