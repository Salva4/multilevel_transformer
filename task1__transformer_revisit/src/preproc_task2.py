import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


fn_data_de_tr = '/Users/marcsalvado/Desktop/SCRIPTS/4-MLT/ML_PQ/data/task2_ende/train.de.txt'
fn_data_en_tr = '/Users/marcsalvado/Desktop/SCRIPTS/4-MLT/ML_PQ/data/task2_ende/train.en.txt'
fn_data_de_te = '/Users/marcsalvado/Desktop/SCRIPTS/4-MLT/ML_PQ/data/task2_ende/newstest2015.de.txt'
fn_data_en_te = '/Users/marcsalvado/Desktop/SCRIPTS/4-MLT/ML_PQ/data/task2_ende/newstest2015.en.txt'
fn_voc_de = '/Users/marcsalvado/Desktop/SCRIPTS/4-MLT/ML_PQ/data/task2_ende/vocab.50K.de.txt'
fn_voc_en = '/Users/marcsalvado/Desktop/SCRIPTS/4-MLT/ML_PQ/data/task2_ende/vocab.50K.en.txt'


def data_(fn_data, voc):
    sentences_enc = []
    idx_unk = voc['<unk>']
    max_len = -1

    with open(fn_data, 'r') as f:
        for i, line in enumerate(f):
            if i%100000 == 0: print('extracting', i)
            line = line.strip().split()
            sentence_enc = [voc.get(word, idx_unk) for word in line]
            sentences_enc.append(sentence_enc)
            max_len = max(max_len, len(sentence_enc))

    idx_pad = voc['<pad>']
    data = idx_pad * torch.ones(size=(len(sentences_enc), max_len), 
                                dtype=torch.int32)

    for i, sentence_enc in enumerate(sentences_enc):
        if i%100000 == 0: print('preparing', i)
        data[i, :len(sentence_enc)] = torch.tensor(sentence_enc, 
                                                   dtype=torch.int32)

    return data


def main():
    voc_de = voc_(fn_voc_de)
    voc_en = voc_(fn_voc_en)

    data_de_tr = data_(fn_data_de_tr, voc_de)
    data_en_tr = data_(fn_data_en_tr, voc_en)
    data_de_te = data_(fn_data_de_te, voc_de)
    data_en_te = data_(fn_data_en_te, voc_en)

    return (voc_de, voc_en), \
           (data_de_tr, data_en_tr, data_de_te, data_en_te)


def voc_(fn_voc):
    voc = {'<pad>': 0}

    with open(fn_voc, 'r') as f:
        for i, line in enumerate(f):
            word = line.strip()
            voc[word] = i+1

    return voc

































