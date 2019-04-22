#-*- coding: utf-8 -*-

import os, sys
import time
import datetime
import numpy as np
import tensorflow as tf
from optparse import OptionParser
##
from rnn import RNN
## import function
from data import load_xy, batch_iter, BucketedDataIterator
from common import get_tfconfig
from params import BATCH_SIZE

from pandas import DataFrame
import pandas as pd

##
NUM_CLASSES=2
HIDDEN_SIZE=300
EMBEDDING_DIM = 300 
KEEP_PROB = 0.5
L2_REG_LAMBDA = 0.0
BATCH_SIZE = 10
## 
NUM_CHECKPOINTS = 5
EVAL_EVERY = 100
CHECKPOINT_EVERY = 100
NUM_EPOCHS = 200
    
embedding_dir = './word2vec/'
EMBEDDING_DIMS = 300
MAX_NB_WORDS = 200000000000

# set load_xy in preprocess
max_document_length = None

def load_word2vec(vocab_processor):
    vocab_dict = vocab_processor.vocabulary_._mapping
    nb_words = min(MAX_NB_WORDS, len(vocab_dict))
    filename = os.path.join(embedding_dir, 'GoogleNews-vectors-negative300.bin')
    import gensim
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIMS))
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    print 'found {} word vectors.'.format(len(wv_model.vocab))
    for word, i in vocab_dict.items():
        if word in wv_model.vocab:
            embedding_matrix[i] = wv_model[word]
    return embedding_matrix 

def get_indexs(vocab_dict, tokens):
    indexs = []
    for token in tokens:
        if token in vocab_dict:
            indexs.append(vocab_dict[token])
        else:
            indexs.append(vocab_dict["<UNK>"])
    return indexs

def preprocess():
    print("load data")
    x_text, y, max_document_length = load_xy()
    print(len(x_text), len(y), max_document_length)
    ##
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor.fit_transform(x_text) ## 데이터 집어넣고
    vocab_dict = vocab_processor.vocabulary_._mapping # word:idx
    print("vocab size : {}".format(len(vocab_processor.vocabulary_)))
    # x_text = [ sentence1, sentence2 ... ]
    # y = [ sentence1 class name, sentence2 class name, ... ]
    print("formatting data")
    datas = {'class':[], 'sentence':[], 'as_numbers':[], 'length':[]}
    for idx, x_sentence in enumerate(x_text):
        tokens = x_sentence.strip().split()
        as_numbers = get_indexs(vocab_dict, tokens)
        datas['class'].append(y[idx])
        datas['sentence'].append(x_sentence)
        datas['as_numbers'].append(as_numbers)
        datas['length'].append(len(as_numbers))
    df = DataFrame(datas)
    # split train/test set
    train_len, test_len = np.floor(len(df)*0.9), np.floor(len(df)*0.1)
    ##
    print("load embedding matrix")
    embedding_matrix = None #load_word2vec(vocab_processor)
    train_data, test_data = df.ix[:train_len-1], df.ix[train_len:train_len + test_len]
    return train_data, vocab_processor, test_data, embedding_matrix

def train(train_data, vocab_processor, test_data, embedding_matrix, iterator = BucketedDataIterator):
    print("train")
    config = get_tfconfig()
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            with sess.as_default():
                gstep = tf.Variable(0, name="gstep", trainable=False)
                ## build graph
                vocab_size = len(vocab_processor.vocabulary_)
                rnn = RNN(BATCH_SIZE, vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_CLASSES)
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                grads_vars = optimizer.compute_gradients(rnn.loss)
                train_op = optimizer.apply_gradients(grads_vars, global_step=gstep)
                ## output dictionary for models
                time_stamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", time_stamp))
                print("writing to {}".format(out_dir))
                ## gradients summaries
                #grad_summaries = []
                #for g, v in grads_vars:
                #    if g:
                #        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                #        grad_summaries.append(grad_hist_summary)
                #        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                #        grad_summaries.append(sparsity_summary)
                #grad_summaries_merged = tf.summary.merge(grad_summaries)
                #train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                s_loss = tf.placeholder(tf.float32)
                s_acc = tf.placeholder(tf.float32)
                loss_summary = tf.summary.scalar("loss", s_loss) #rnn.loss)
                acc_summary = tf.summary.scalar("accuracy", s_acc) #rnn.accuracy)
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                ## define summary path 
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
                ## saver
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)
                vocab_processor.save(os.path.join(out_dir, "vocab"))
                ## start session
                accuracy_list = []
                sess.run(tf.global_variables_initializer())
                def train_step(batch):
                    feed_dict = {rnn.input_x : batch[0], rnn.input_y : batch[1], rnn.sequence_length : batch[2], rnn.dropout_keep_prob : KEEP_PROB}
                    _, loss, accuracy = sess.run([train_op, rnn.loss, rnn.accuracy], feed_dict)
                    step, summaries = sess.run([gstep, train_summary_op], {s_loss:loss, s_acc:accuracy})
                    #_, step, summaries, loss, accuracy = sess.run([train_op, gstep, train_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    if step % 10 == 0:
                        print("train {}: step {}. loss {}, acc {}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)
                def dev_step(batch, writer=None):
                    feed_dict = {rnn.input_x : batch[0], rnn.input_y : batch[1], rnn.sequence_length : batch[2], rnn.dropout_keep_prob : 1.0}
                    #step, summaries, loss, accuracy = sess.run([gstep, dev_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                    loss, accuracy = sess.run([rnn.loss, rnn.accuracy], feed_dict)
                    #print("dev {}: step {}. loss {}, acc {}".format(time_str, step, loss, accuracy))
                    return loss, accuracy
                def check_stop():
                    if len(accuracy_list) > 6:
                        max_acc = max(accuracy_list)
                        mm = 5
                        for idx, iacc in enumerate(accuracy_list[-5:]):
                            if max_acc > iacc: mm-=1
                        if mm == 0: 
                            return True
                    return False
                ## define iterator
                tr = iterator(train_data)
                te = iterator(test_data)
                cur_epoch = 0
                num_epochs = 100
                while cur_epoch < num_epochs:
                    batch = tr.next_batch(BATCH_SIZE)
                    train_step(batch)
                    cur_step = tf.train.global_step(sess, gstep)
                    if tr.epochs > cur_epoch:
                        cur_epoch += 1 
                    if cur_step % EVAL_EVERY == 0: 
                        print("\nEvaluation : ")
                        te_epoch = te.epochs
                        tot_loss = []
                        tot_acc = []
                        while te.epochs == te_epoch:
                            dev_batch = te.next_batch(BATCH_SIZE)
                            loss, acc = dev_step(dev_batch) #, writer=dev_summary_writer)
                            tot_loss.append(loss)
                            tot_acc.append(acc)
                        avg_loss = np.mean(tot_loss)
                        avg_acc = np.mean(tot_acc)
                        step, summaries = sess.run([gstep, dev_summary_op], {s_loss:avg_loss, s_acc:avg_acc})
                        if dev_summary_writer:
                            dev_summary_writer.add_summary(summaries, step)
                        ##
                        time_str = datetime.datetime.now().isoformat()
                        print("dev {}: step {}. avg loss {}, avg acc {}".format(time_str, step, avg_loss, avg_acc))
                        ##
                        max_acc = max(accuracy_list) if len(accuracy_list) > 2 else -1000
                        if max_acc < avg_acc:
                            print("update {} -> {}, length {}".format(max_acc, avg_acc, len(accuracy_list)))
                            accuracy_list.append(avg_acc)
                        print("")
                        ####
                        path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        if check_stop():
                            break
def main():
    train_data, vocab_processor, test_data, embedding_matrix = preprocess()
    train(train_data, vocab_processor, test_data, embedding_matrix)
    
if __name__ == '__main__':
    main()

