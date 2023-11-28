import datasets
import os
from torch.utils.data import DataLoader
import tqdm

# src_language, tgt_language = 'de', 'en'
src_language, tgt_language = 'en', 'de'
dir_data = os.path.join('..', 'data', 'deen_translation')
if 'data' not in os.listdir('..'): dir_data = '../' + dir_data  # try commenting this
fn = {
  'training': {
    'src': 'train.y',#'train.x', 
    'tgt': 'train.x',#'train.y'
  }, 
  'validation': {
    'src': 'interpolate.y',#'interpolate.x', 
    'tgt': 'interpolate.x',#'interpolate.y',
  },
}

def obtain_data(_vars):
  _vars.splits = ['training', 'validation']
  data_sets = {split: {'translation': []} for split in _vars.splits}  # init
  
  for split in _vars.splits:  # extract sentences from files 
    path_src = open(os.path.join(dir_data, fn[split]['src']), 'r')
    path_tgt = open(os.path.join(dir_data, fn[split]['tgt']), 'r')

    print(f'Obtaining "{split}" data')
    for i, (line_src, line_tgt) in enumerate(tqdm.tqdm(zip(path_src, path_tgt))):
      if _vars.debug and i > 2000: break
      data_sets[split]['translation'].append(
        {
          src_language: line_src.strip(),
          tgt_language: line_tgt.strip(),
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
    'training': DataLoader(
      data_set_dicts['training'], 
      batch_size=_vars.batch_size, shuffle=True,
    ),
    'validation': DataLoader(
      data_set_dicts['validation' ], 
      batch_size=_vars.batch_size, shuffle=False,
    )
  }

  _vars.src_language, _vars.tgt_language = src_language, tgt_language
  _vars.data_sets, _vars.data_loaders = data_set_dicts, data_loaders

## Adapted from https://huggingface.co/docs/transformers/tasks/translation
def φ_preprocess(partition, tokenizer):
  inputs  = [instance[src_language] for instance in partition['translation']]
  targets = [instance[tgt_language] for instance in partition['translation']]
  model_inputs = tokenizer(
    inputs, text_target=targets, max_length=128, padding='max_length', 
    # truncation=True,
  )
  return model_inputs




