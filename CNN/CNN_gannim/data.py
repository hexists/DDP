#-*- coding: utf-8 -*-
import re
import numpy as np

DATA_PATH = './data'
FILE_PREFIX = 'rt-polarity'
POS_CLASS_NAME = 'pos'
NEG_CLASS_NAME = 'neg'

def read_data(file_name):
    with open(file_name, 'r') as ff:
        return [line.strip() for line in ff.readlines()]

def load_xy():
    pos_data_file = "{}/{}.{}".format(DATA_PATH, FILE_PREFIX, POS_CLASS_NAME)
    neg_data_file = "{}/{}.{}".format(DATA_PATH, FILE_PREFIX, NEG_CLASS_NAME)
    pos_exams = read_data(pos_data_file)
    neg_exams = read_data(neg_data_file)
    inputs = [clean_str(exam) for exam in pos_exams + neg_exams]
    # [NEG, POS]
    pos_labels = [[0, 1] for _ in pos_exams]
    neg_labels = [[1, 0] for _ in neg_exams]
    y = np.concatenate([pos_labels, neg_labels], 0)
    return [inputs, y, max([len(exam.split(' ')) for exam in inputs])]
    
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            suffled_data = data[shuffle_indices]
        else:
            suffled_data = data
        for bnum in range(num_batches_per_epoch):
            sidx = bnum * batch_size
            eidx = min((bnum+1)*batch_size, data_size)
            yield suffled_data[sidx:eidx]
    
