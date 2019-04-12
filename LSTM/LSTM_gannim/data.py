#-*- coding: utf-8 -*-
import re
import numpy as np
from pandas import DataFrame
import pandas as pd

DATA_PATH = './data'
FILE_PREFIX = 'rt-polarity'
POS_CLASS_NAME = 'pos'
NEG_CLASS_NAME = 'neg'


class SimpleDataIterator():
    def __init__(self, df):
        self.df = df 
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.sursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.sursor+n-1]
        self.cursor += n
        return x, res['class'], res['length']

class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.sursor+n-1]
        self.cursor += n

        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['class'], res['length']

class BucketedDataIterator():
    def __init__(self, df, num_buckets = 5):
        df = df.sort_values('length').reset_index(drop=True)
        #print df
        self.size = len(df) / num_buckets 
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.ix[bucket*self.size:(bucket+1)*self.size - 1])
        self.num_buckets = num_buckets 
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()
        self.epochs = 0

    def shuffle(self):
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0
        #print self.dfs

    def next_batch(self, n):
        #print(" === next_batch === ")
        if np.any(self.cursor+n+1 > self.size):
            print(" === suffling === ")
            self.epochs += 1
            self.shuffle()
        ci = np.random.randint(0, self.num_buckets)
        # ix : integer position과 label 모두 사용할 수 있다. 만약 label이 숫자라면 label-based indexing이 된다.
        res = self.dfs[ci].ix[self.cursor[ci]:self.cursor[ci]+n-1]
        res_idxs = [i for i in range(self.cursor[ci], self.cursor[ci]+n)]
        self.cursor[ci] += n
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            res_idx = res_idxs[i]
            x_i[:res['length'][res_idx]] = np.array(res['as_numbers'][res_idx])
            if 18759 in np.array(res['as_numbers'][res_idx]):
                print ">>>>>>>>>>>>>>>>>>>>>"
                print res['as_numbers'][res_idx]
        y = np.zeros([n, 2], dtype=np.int32)
        for i, y_i in enumerate(y):
            res_idx = res_idxs[i]
            y_i[:2]= np.array(res['class'][res_idx])
        l = np.zeros([n], dtype=np.int32)
        l[:n] = np.array(res['length'])
        #print x
        #print y
        #print l
        return x, y, l

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
    
