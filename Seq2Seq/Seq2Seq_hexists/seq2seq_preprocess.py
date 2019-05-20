#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np


START_SYMBOL = '^'
END_SYMBOL = '$'
PAD_SYMBOL = '+'
UNK_SYMBOL = '?'
SYMBOLS = {PAD_SYMBOL: 0, END_SYMBOL: 1, UNK_SYMBOL: 2, START_SYMBOL: 3}


def load_data(path):
    srcs, tgts = [], []

    with open(path) as fp:
        for buf in fp:
            line = buf.rstrip().split('\t')
            if len(line) != 2:
                continue
            src, tgt = line
            srcs.append(src)
            tgts.append(tgt)
    return srcs, tgts


def create_lookup_tables(texts):
    voc2idx = {}

    vocab = {v for text in texts for v in text}

    for vi, v in enumerate(SYMBOLS):
        voc2idx[v] = vi
    for vi, v in enumerate(vocab, len(SYMBOLS)):
        voc2idx[v] = vi

    idx2voc = {vi: v for v, vi in voc2idx.items()}

    return voc2idx, idx2voc


def text2idx(voc2idx, texts, t_type):

    text_ids, max_len = [], 0

    for text in texts:
        if t_type == 'inp':
            ids = [voc2idx[char] for char in text]
        elif t_type == 'out':
            ids = [voc2idx[char] for char in (START_SYMBOL + text)]
        elif t_type == 'tgt':
            ids = [voc2idx[char] for char in (text + END_SYMBOL)]
        max_len = len(ids) if max_len < len(ids) else max_len
        text_ids.append(ids)
    return np.array(text_ids), max_len


def preprocess_data():
    path = './data/transliteration/data.txt'
    # load_data
    inps, outs = load_data(path)
    # create lookup table
    inp_voc2idx, inp_idx2voc = create_lookup_tables(inps)
    out_voc2idx, out_idx2voc = create_lookup_tables(outs)
    # create text to idx
    inp_ids, inp_max_len = text2idx(inp_voc2idx, inps, t_type='inp')
    out_ids, out_max_len = text2idx(out_voc2idx, outs, t_type='out')
    tgt_ids, tgt_max_len = text2idx(out_voc2idx, outs, t_type='tgt')
    # save data
    pickle.dump((
        (inp_ids, out_ids, tgt_ids),
        (inp_max_len, out_max_len, tgt_max_len),
        (inp_voc2idx, out_voc2idx),
        (inp_idx2voc, out_idx2voc)), open('./pickle/seq2seq.pickle', 'wb'))

if __name__ == '__main__':
    preprocess_data()
