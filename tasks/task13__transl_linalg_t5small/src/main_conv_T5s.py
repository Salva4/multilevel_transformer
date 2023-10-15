# Author: Marc Salvadó Benasco

############################## IMPORTS
import os
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
# import matplotlib.pyplot as plt
import time

# colored: to print the losses and accuracies related to the training data in 
#...blue and the ones related to the validation data in red.
colour_imported = True
try:
  from termcolor import colored   
except: 
  colour_imported = False

from torchtext.data.utils import get_tokenizer
from torchtext.data.metrics import bleu_score

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_lays_enc', type=str, default='4')
parser.add_argument('--n_lays_dec', type=str, default='4')
args = parser.parse_args()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-small')
######################################


##################### GLOBAL VARIABLES
DEVICE = 'cuda'

DATASET_DIR = "../data/"

TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"
######################################


################## CLASSES & FUNCTIONS
class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>',
                 sos_token='<sos>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1
        
        # add the default unknown token
        self.id_to_string[2] = eos_token
        self.string_to_id[eos_token] = 2   

        # add the default unknown token
        self.id_to_string[3] = sos_token
        self.string_to_id[sos_token] = 3

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.sos_id = 3

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if extend_vocab is True, add the new word
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt files and generate parallel text dataset:
# self.data[idx][0] is the tensor of source sequence
# self.data[idx][1] is the tensor of target sequence
# See examples in the cell below.
class ParallelTextDataset(Dataset):

    def __init__(self, src_file_path, tgt_file_path, src_vocab=None,
                 tgt_vocab=None, extend_vocab=False, device='cuda'):
        (self.data, self.src_vocab, self.tgt_vocab, self.src_max_seq_length,
         self.tgt_max_seq_length) = self.parallel_text_to_data(
            src_file_path, tgt_file_path, src_vocab, tgt_vocab, extend_vocab,
            device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def parallel_text_to_data(self, src_file, tgt_file, src_vocab=None,
                              tgt_vocab=None, extend_vocab=False,
                              device='cuda'):
        # Convert paired src/tgt texts into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file.

        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        if src_vocab is None:
            src_vocab = Vocabulary()

        if tgt_vocab is None:
            tgt_vocab = Vocabulary()
        
        data_list = []
        # Check the max length, if needed construct vocab file.
        src_max = 0
        with open(src_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]  # remove line break
                length = len(tokens)
                if src_max < length:
                    src_max = length

        tgt_max = 0
        with open(tgt_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]
                length = len(tokens)
                if tgt_max < length:
                    tgt_max = length
        tgt_max += 2  # add for begin/end tokens
                    
        src_pad_idx = src_vocab.pad_id
        tgt_pad_idx = tgt_vocab.pad_id

        tgt_eos_idx = tgt_vocab.eos_id
        tgt_sos_idx = tgt_vocab.sos_id

        # Construct data
        src_list = []
        print(f"Loading source file from: {src_file}")
        with open(src_file, 'r') as text:
            for i, line in enumerate(tqdm(text)):
              if args.debug and i == 10000: break
              seq = []
              tokens = list(line)[:-1]
              for token in tokens:
                  seq.append(src_vocab.get_idx(
                      token, extend_vocab=extend_vocab))
              var_len = len(seq)
              var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
              # padding
              new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
              new_seq[:var_len] = var_seq
              src_list.append(new_seq)

        tgt_list = []
        print(f"Loading target file from: {tgt_file}")
        with open(tgt_file, 'r') as text:
            for i, line in enumerate(tqdm(text)):
              if args.debug and i == 10000: break
              seq = []
              tokens = list(line)[:-1]
              # append a start token
              seq.append(tgt_sos_idx)
              for token in tokens:
                  seq.append(tgt_vocab.get_idx(
                      token, extend_vocab=extend_vocab))
              # append an end token
              seq.append(tgt_eos_idx)

              var_len = len(seq)
              var_seq = torch.tensor(seq, device=device, dtype=torch.int64)

              # padding
              new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
              new_seq[:var_len] = var_seq
              tgt_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(src_list) == len(tgt_list)
        for i in range(len(src_list)):
            data_list.append((src_list[i], tgt_list[i]))

        print("Done.")
            
        return data_list, src_vocab, tgt_vocab, src_max, tgt_max


########
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        assert x.size(0) < self.max_len, (
            f"Too long sequence length: increase `max_len` of pos encoding")
        # shape of x (len, B, dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 3. Model
# 3.1 Implementation of a Transformer encoder-decoder model (directly via the 
#...submodules that are called inside nn.Transformer)
class Trans(nn.Module):
  def __init__(
      self, num_layers_encoder, num_layers_decoder, dim_hidden, dim_ff, 
      attention_heads, dropout, vocabulary_source, vocabulary_target,
  ):
    super(Trans, self).__init__()

    # Constants
    dim_alphabet_source = len(vocabulary_source)
    dim_alphabet_target = len(vocabulary_target)
    self.vocabulary_source = vocabulary_source
    self.vocabulary_target = vocabulary_target
    self.dim_hidden = dim_hidden

    # Embedding & Positional encoding
    self.embedding_encoder = nn.Embedding(dim_alphabet_source, dim_hidden)
    self.embedding_decoder = nn.Embedding(dim_alphabet_target, dim_hidden)
    self.positional_encoder = PositionalEncoding(dim_hidden)

    # Encoder
    layer_encoder = nn.TransformerEncoderLayer(
        d_model = dim_hidden, 
        nhead = attention_heads, 
        dim_feedforward = dim_ff,
        dropout = dropout,
        batch_first = False,
        norm_first = True
    )
    self.encoder = nn.TransformerEncoder(layer_encoder, num_layers_encoder)

    # Decoder
    layer_decoder = nn.TransformerDecoderLayer(
        d_model = dim_hidden,
        nhead = attention_heads,
        dim_feedforward = dim_ff,
        dropout = dropout,
        batch_first = False,
        norm_first = True
    )
    self.decoder = nn.TransformerDecoder(layer_decoder, num_layers_decoder)

    # Classification (Linear) layer
    self.fc = nn.Linear(dim_hidden, dim_alphabet_target, bias=True)


  def forward(self, source, target_input):
    memory, mask_padding_source = self.encode(source)   # encoder
    mask_padding_memory = mask_padding_source.clone()   # memory mask
    scores = self.decode(target_input, memory, mask_padding_memory)   # decoder
    return scores


  def encode(self, source):
    # source has shape (batch, len_max_source)

    mask_padding_source = (source == self.vocabulary_source.pad_id)    
                                                      # (batch, len_max_source)
    source = source.transpose(0, 1)   # (len_max_source, batch)

    source_embedded = self.embedding_encoder(source)    
                                          # (len_max_source, batch, dim_hidden)
    source_posEncoding = self.positional_encoder(source_embedded)   
                                          # (len_max_source, batch, dim_hidden)
    memory = self.encoder(
        src = source_posEncoding, 
        src_key_padding_mask = mask_padding_source
    )                                     # (len_max_source, batch, dim_hidden)
    
    return memory, mask_padding_source


  def decode(self, target_input, memory, mask_padding_memory):
    # target_input has shape (batch, len_max_target-1)

    mask_attention_target_input = nn.Transformer.generate_square_subsequent_mask(
        sz = target_input.size()[1]
    ).to(DEVICE)    # (len_max_target-1, len_max_target-1)
    mask_padding_target_input = (target_input == self.vocabulary_target.pad_id)    
                                                    # (batch, len_max_target-1)

    target_input = target_input.transpose(0, 1)   # (len_max_target-1, batch)
    target_input_embedded = self.embedding_decoder(target_input)    
                                        # (len_max_target-1, batch, dim_hidden)
    target_input_posEncoding = self.positional_encoder(target_input_embedded)   
                                        # (len_max_target-1, batch, dim_hidden)

    output = self.decoder(
        tgt = target_input_posEncoding,
        memory = memory,
        tgt_mask = mask_attention_target_input,
        tgt_key_padding_mask = mask_padding_target_input,
        memory_key_padding_mask = mask_padding_memory
    )                                   # (len_max_target-1, batch, dim_hidden)

    scores = self.fc(output.transpose(0, 1))    
                                # (batch, len_max_target-1, dim_alphabet_target)
    return scores


# 4. Greedy search
def greedy(model, source, length, vocabulary_target):
  target_input = torch.full_like(source, vocabulary_target.sos_id)[:, 0:1]
  eosed = torch.full_like(target_input, False)

  # 4.1 --> encoder forwarded once, decoder consecutively
  # Encoder: only once entirely
  memory, mask_padding_source = model.encode(source)
  mask_padding_memory = mask_padding_source.clone()

  # Decoder: multiple forwards
  greedy_outputs = None
  batch_size = target_input.size()[0]

  # 4.3 Stopping criteria. Part 1/2: if by the maximum length of the targets,
  #...the greedy search hasn't outputted a "<eos>", then the answer will be 
  #...wrong independently of how it would have continued.
  for i in range(length):   
    outputs = model.decode(target_input, memory, mask_padding_memory)[:, -1:, :]
    predictions = outputs.argmax(dim=2)
    target_input = torch.cat((target_input, predictions), dim = 1)
    greedy_outputs = torch.cat(
        (greedy_outputs, outputs), 
        dim=1
    ) if greedy_outputs != None else outputs

    # eosed (<eos>-ed) shows whether the greedy search has outputted <eos>, at 
    #...some time, for each sentence.
    eosed = torch.logical_or(
        eosed,
        predictions == vocabulary_target.eos_id
    )

    # 4.3 Stopping criteria. Part 2/2: if the greedy search has at some point 
    #...outputted an <eos> for each sentence, then finish the search.
    if eosed.all():
      break

  # 4.4 Batch mode. Some sentences are forwarded to the decoder despite having
  #...already outputted an <eos>. This is because it is faster to decode in
  #...batch mode, but the "extra outputs" (after <eos>) will be filtered later. 

  predictions = target_input[:, 1:]   # Subtract the <sos>
  return greedy_outputs, predictions


# 6. Training
# 6.2 (Implement ...)
def train(model, optimizer, loss_function, train_data_loader, valid_data_loader,
          vocabulary_source, vocabulary_target, time_limit_minutes=1e9):
  print('Begin training')
  t0 = time.time()

  # Constants
  n_gradients_update = 10   # (for gradients accumulation)
  n_monitoring = 500*10
  n_early_stop = 10
  min_accuracy_early_stop = .95
  iter_valid_data_loader = iter(valid_data_loader)

  # Initialise variables
  counter_gradients_update = 0
  counter_monitoring = 0
  correct_training = 0 
  total_training = 0
  counter_accuracyValid_notBetter = 0
  max_accuracyValid = 0.
  losses_training = []
  losses_validation = []
  losses_temp_training = []
  accuracies_training = []
  accuracies_validation = []
  first_loss = True
  finished = False
  optimizer.zero_grad()

  ### Load model & history (only in algebra__linear_1d)
  '''
  try:
    checkpoint = torch.load(
      'drive/MyDrive/Colab Notebooks/2 - USI DLL/models/state_ex8_alg1.pt'
    )
  except:
    checkpoint = torch.load(
      'drive/MyDrive/Colab Notebooks/2 - USI DLL/models/state_ex8_alg2.pt'
    )
  model.load_state_dict(checkpoint['model_state'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  losses_training = checkpoint['losses_training']
  losses_validation = checkpoint['losses_validation']
  accuracies_training = checkpoint['accuracies_training']
  accuracies_validation = checkpoint['accuracies_validation']
  '''
  #####################################################

  while not finished:
    model.train()   # train
    for batch in train_data_loader:
      sources, targets = batch
      targets_input, targets_output = targets[:, :-1], targets[:, 1:]   # 6.2

      # Forward
      outputs = model(sources, targets_input)   
                                            # (batch, len_max_targets, alphabet)
      outputs_alphabet_dim1 = outputs.transpose(1, 2)   
                                            # (batch, alphabet, len_max_targets)
      loss = loss_function(outputs_alphabet_dim1, targets_output)
      predictions = outputs.argmax(dim=2)

      # Backward
      loss.backward()
      counter_gradients_update += 1
      if counter_gradients_update % n_gradients_update == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)  # Grad. clipping
        optimizer.step()    # 6.3 Gradient accumulation
        optimizer.zero_grad()

      correct_training += torch.logical_or(
          predictions == targets_output, 
          targets_output == vocabulary_target.pad_id
      ).prod(dim=1).sum(dim=0).item()
      total_training += targets_output.size()[0]

      # Monitoring
      losses_temp_training.append(loss.item())
      counter_monitoring += 1
      if first_loss:
        text = f'first training loss {loss.item() : .4e}'
        print(colored(text, 'blue') if colour_imported else text)
        print()
        first_loss = False

      if args.debug: print(loss.item())

      if counter_monitoring % n_monitoring == 0:
        model.eval()   # evaluate 
        with torch.no_grad():        
          # (I) training set
          losses_training.append(np.mean(losses_temp_training))
          accuracies_training.append(correct_training / total_training)
          correct_training, total_training = 0, 0

          text = f'training loss {loss.item() : .4e}'
          text += f'\ttraining accuracy {100 * accuracies_training[-1] : .4f}%'
          print(colored(text, 'blue') if colour_imported else text)

          # We'll take as an example the first entry in the batch
          print_example(predictions, sources, targets_output, vocabulary_source, 
                       vocabulary_target, 'blue')

          losses_temp_validation = []
          correct_validation, total_validation = 0, 0
          candidate_corpus, reference_corpus = [], []
          for _ in range(50):
            # (II) validation set
            batch = next(iter_valid_data_loader, None)
            if batch == None:
              iter_valid_data_loader = iter(valid_data_loader)
              batch = next(iter_valid_data_loader)

            sources, targets = batch
            targets_output = targets[:, 1:]

            # Forward
            # Greedy
            outputs, predictions = greedy(
                model, 
                sources, 
                targets.size()[1]-1, 
                vocabulary_target
            )   # outputs:     (batch, <eos>-ed length, alphabet)
                # predictions: (batch, <eos>-ed length)
            targets_output = targets_output[:, :predictions.size()[1]]
                                                        # (batch, <eos>-ed length)
            outputs_alphabet_dim1 = outputs.transpose(1, 2)   
                                              # (batch, alphabet, <eos>-ed length)
            loss = loss_function(outputs_alphabet_dim1, targets_output)
            losses_temp_validation.append(loss.item())

            correct_validation += torch.logical_or(
                predictions == targets_output, 
                targets_output == vocabulary_target.pad_id
            ).prod(dim=1).sum(dim=0).item()
            total_validation += targets_output.size()[0]

            for prediction, target_output in zip(predictions, targets_output):
              pred = ''.join([vocabulary_target.id_to_string[i.item()] \
                                                 for i in prediction])
              tgt = ''.join([vocabulary_target.id_to_string[i.item()] \
                                              for i in target_output])
              for char in '.,;:!?':
                pred = pred.replace(char, ' '+char)
                tgt = tgt.replace(char, ' '+char)
              pred = pred.replace('<eos>', ' <eos>')
              tgt = tgt.replace('<eos>', ' <eos>')

              candidate_corpus.append(pred.split())
              reference_corpus.append([tgt.split()])

          # 5. Accuracy computation
          # 5.1 (Also above in training set)
          accuracy_validation = correct_validation / total_validation

          # Monitoring
          # losses_validation.append(loss.item())
          losses_validation.append(np.mean(losses_temp_validation))
          accuracies_validation.append(accuracy_validation)

          text = f'validation loss {loss.item() : .4e}'
          text += f'\tvalidation accuracy {100*accuracies_validation[-1] :.4f}%'
          print(colored(text, 'red') if colour_imported else text)
          print_example(predictions, sources, targets_output, vocabulary_source, 
                       vocabulary_target, 'red')
          print(f'val bleu {bleu_score(candidate_corpus, reference_corpus)}')
          
          # Early stop: 
          if accuracy_validation > max_accuracyValid:
            max_accuracyValid = accuracy_validation
            counter_accuracyValid_notBetter = 0
          
          else:
            counter_accuracyValid_notBetter += 1
            condition1 = (counter_accuracyValid_notBetter >= n_early_stop)
                          # if valid. accuracy has not reached a global maximum 
                          #...within the last n_early_stop monitoring steps
            condition2 = ( min(
                accuracies_validation[-n_early_stop:]
            ) >= min_accuracy_early_stop )
                          # And all of the last n_early_stop monitoring steps 
                          #...are above min_accuracy_early_stop (only because
                          #...it is known that it will reach this accuracy)
            if condition1 and condition2:
              text = 'Early stopping: validation accuracy has not improved in '
              text += f'the last {n_early_stop} monitoring steps and the '
              text += 'minimum of these accuracies is above '
              text += f'{100 * min_accuracy_early_stop}%'
              print(text)
              finished = True   # Then, early stop. 
              break

          ### Save model & history (only in algebra__linear_1d)
          '''
          model_state = {
              'model_state': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'losses_training': losses_training,
              'losses_validation': losses_validation,
              'accuracies_training': accuracies_training,
              'accuracies_validation': accuracies_validation
          }
          torch.save(
            model_state, 
            'drive/MyDrive/Colab Notebooks/2 - USI DLL/models/state_ex8_alg1.pt'
          )
          torch.save(
            model_state, 
            'drive/MyDrive/Colab Notebooks/2 - USI DLL/models/state_ex8_alg2.pt'
          ) # save it in two files so that if it gets interrupted while saving
            #...one file (so the file becomes blank), the model can be recovered
            #...through the other file.
          '''
          #####################################################

      # Stop because time limit was exceeded.
      if time.time()-t0 >= time_limit_minutes * 60:
        print(f'Time limit of {time_limit_minutes} minutes has been exceeded.')
        finished = True
        break

  return (model, optimizer, losses_training, losses_validation, 
          accuracies_training, accuracies_validation)


def print_example(predictions, sources, targets_output, vocabulary_source, 
                 vocabulary_target, colour):
  question = sources[0,:]
  padding_question = (question == vocabulary_source.pad_id).nonzero()
  index_padding_question = padding_question[0] if len(
      padding_question != 0
  ) else len(question)
  text1 = ''.join(
      [
       vocabulary_source.id_to_string[i.item()] for i in question[
                                                      :index_padding_question]
      ]
  )

  answer_predicted = predictions[0,:]
  eos_answer_predicted = (
      answer_predicted == vocabulary_target.eos_id
  ).nonzero()
  index_eos_answer_predicted = eos_answer_predicted[0] if len(
      eos_answer_predicted != 0
  ) else len(answer_predicted)
  text2 = ''.join(
      [
       vocabulary_target.id_to_string[i.item()] for i in answer_predicted[
                                                :index_eos_answer_predicted+1]
      ]
  )

  answer_correct = targets_output[0,:]
  eos_answer_correct = (answer_correct == vocabulary_target.eos_id).nonzero()
  index_eos_answer_correct = eos_answer_correct[0] if len(
      eos_answer_correct != 0
  ) else len(answer_correct)
  text3 = ''.join(
      [
       vocabulary_target.id_to_string[i.item()] for i in answer_correct[
                                                  :index_eos_answer_correct+1]
      ]
  )
  text4 = bool(torch.logical_or(
      predictions == targets_output, 
      targets_output == vocabulary_target.pad_id
  ).prod(dim=1)[0].item())

  for pretext, text in [('QUESTION:', text1), 
                        ('PREDICTED ANSWER:', text2), 
                        ('CORRECT ANSWER:', text3), 
                        ('CORRECT?', 'Yes.' if text4 else 'No.')]:
    if colour_imported:
      print(colored(f'\t{pretext : >30} {text : <12}', colour))
    
    else:
      print(f'\t{pretext : >30} {text : <12}')

  print('\n\n')

'''
def plot_training(losses_training, losses_validation, accuracies_training, 
                  accuracies_validation):
  fig, axs = plt.subplots(1, 2, figsize=(15, 5))
  axs[0].plot(losses_training, 'b', label='training')
  axs[0].plot(losses_validation, 'r', label='validation')
  axs[1].plot(100 * np.array(accuracies_training), 'b', label='training')
  axs[1].plot(100 * np.array(accuracies_validation), 'r', label='validation')
  axs[0].title.set_text('Evolution of loss along the training process')
  axs[1].title.set_text('Evolution of accuracy along the training process')
  axs[0].set_xlabel('Monitoring steps')
  axs[1].set_xlabel('Monitoring steps')
  axs[0].set_ylabel('Loss function')
  axs[1].set_ylabel('Accuracy (%)')
  axs[1].set_yticks(range(0, 101, 10))
  axs[1].set_ylim(0, 100)
  axs[1].plot([0, len(accuracies_validation)], [90, 90], '--k')
  axs[0].legend()
  axs[1].legend()
  print()
######################################
'''

######################## MAIN FUNCTION
def main(task):
  torch.manual_seed(0)

  de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
  en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

  src_file_path = DATASET_DIR + f"{task}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
  tgt_file_path = DATASET_DIR + f"{task}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

  train_set = ParallelTextDataset(src_file_path, tgt_file_path, 
                                  extend_vocab=True, device=DEVICE)

  # get the vocab
  src_vocab = train_set.src_vocab
  tgt_vocab = train_set.tgt_vocab

  src_file_path = DATASET_DIR + f"{task}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
  tgt_file_path = DATASET_DIR + f"{task}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

  valid_set = ParallelTextDataset(src_file_path, tgt_file_path, 
                                  src_vocab=src_vocab, tgt_vocab=tgt_vocab,
                                  extend_vocab=False, device=DEVICE)

  batch_size = args.batch_size 

  train_data_loader = DataLoader(
      dataset=train_set, batch_size=batch_size, shuffle=True)

  valid_data_loader = DataLoader(
      dataset=valid_set, batch_size=batch_size, shuffle=False)


  # 7.1 Training & hyper-parameters + 7.3 Hyper-parameter tuning
  # Parameters
  dim_hidden = 512#256
  att_heads = 8
  num_enc_layers_list = [int(i) for i in args.n_lays_enc.split('-')]
  num_dec_layers_list = [int(i) for i in args.n_lays_dec.split('-')]
  dim_ff = 2048#1024
  dropout = 0.1   # Value by default in nn.Transformer and its submodules
  learning_rate = args.lr

  assert len(num_enc_layers_list) == len(num_dec_layers_list) 

  for k in range(len(num_enc_layers_list)):
    num_enc_layers, num_dec_layers = num_enc_layers_list[k], num_dec_layers_list[k]

    torch.manual_seed(1)

    # Model
    model = Trans(
        num_enc_layers, num_dec_layers, dim_hidden, dim_ff, att_heads, dropout,
        src_vocab, tgt_vocab
    ).to(DEVICE)
    loss_function = nn.CrossEntropyLoss(ignore_index = tgt_vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    torch.manual_seed(2)

    (model, optimizer, losses_training, losses_validation, accuracies_training,
      accuracies_validation) = train(
        model, 
        optimizer, 
        loss_function, 
        train_data_loader, 
        valid_data_loader, 
        src_vocab, tgt_vocab,
        time_limit_minutes=24/len(num_enc_layers_list)*60,
    )
    
    # plot_training(losses_training, losses_validation, accuracies_training, 
    #               accuracies_validation)
    
  return model
######################################


############################ EXECUTION
# 7.1, 7.4 & 7.5 - Train with different tasks
# Choose your task by commenting/uncommenting the following lines:
#task = 'numbers__place_value'
#task = 'comparison__sort'
task = 'deen_translation'
model = main(task)
######################################

