## Partially taken from Karpathy's github: [url]

import os
import torch
# from transformers import GPT2Tokenizer, GPT2Model  <-- below

DATA_DIR = os.path.join('..', 'data')

def obtain_data(_vars):
  data_path = os.path.join(DATA_DIR, _vars.input_text + '.txt')

  print('1.1 Reading text')#, end='... ')
  with open(data_path, 'r', encoding='utf-8') as f:
    if not _vars.debug:
      text = f.read()
    else: 
      text = ''
      for j, i in enumerate(f):
        if j > 10: break
        text += i
  print('-> Done.')#'Done')

  if _vars.tokenization == 'character':
    print('1.2 Building character-level tokenizer')#, end='... ')
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocabulary_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    print('Done')

    print('1.3 Encoding data', end='... ')
    data_set = torch.tensor(encode(text), dtype=torch.long)
    print('-> Done.')#'Done')

  elif _vars.tokenization == 'gpt2':
    print('1.2.1 Downloading the GPT2-tokenizer')#, end='... ')
    from transformers import GPT2Tokenizer#, GPT2Model
    print('-> Done.')#'Done')

    print('1.2.2 Obtaining gpt2 tokenizer')#, end='... ')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer.max_length = torch.inf
    # tokenizer.pad_token = '<pad>'
    decode = tokenizer.decode
    vocabulary_size = tokenizer.vocab_size
    print('-> Done.')#'Done')

    print('1.3 Encoding data')#, end='... ')
    data_set = tokenizer(text, max_length=None, return_tensors='pt')['input_ids'][0]
    print('-> Done.')#'Done')

  else: raise Exception()

  print('1.4 Splitting data into training and validation data')#, end='... ')
  n = int(.9*len(data_set))
  training_data_set, validation_data_set = data_set[:n], data_set[n:]
  print('-> Done.')#'Done')

  _vars.data_sets = {
    'training': training_data_set, 'validation': validation_data_set,
  }
  _vars.decoding_function = decode
  _vars.vocabulary_size = vocabulary_size

  return
















