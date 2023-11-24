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
  _vars.splits = ['train', 'test']
  data_sets = {split: {'translation': []} for split in _vars.splits}  # init
  
  for split in _vars.splits:  # extract sentences from files 
    path_src = open(os.path.join(dir_data, fn[split]['src']), 'r')
    path_tgt = open(os.path.join(dir_data, fn[split]['tgt']), 'r')

    print(f'Obtaining "{split}" data')
    for i, (line_src, line_tgt) in enumerate(tqdm.tqdm(zip(path_src, path_tgt))):
      if _vars.debug and i > 2000: break
      data_sets[split]['translation'].append(
        {
          lang_src: line_src.strip(),
          lang_tgt: line_tgt.strip(),
        }
      )

  ## dict --> Dataset --> DatasetDict + tokenize
  data_set_dicts = datasets.DatasetDict({
    split: datasets.Dataset.from_dict(data_sets[split]) \
    for split in _vars.splits
  })
  data_set_dicts = data_set_dicts.map(
    lambda partition: φ_preprocess(partition, _vars.tokenizer), 
    batched=True
  ).with_format('torch', device=_vars.device)

  data_loaders = {
    'train': DataLoader(
      data_set_dicts['train'], batch_size=_vars.batch_size, shuffle=True ,
    ),
    'test': DataLoader(
      data_set_dicts['test' ], batch_size=_vars.batch_size, shuffle=False,
    )
  }

  _vars.lang_src, _vars.lang_tgt = lang_src, lang_tgt
  _vars.data_sets, _vars.data_loaders = data_set_dicts, data_loaders

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




