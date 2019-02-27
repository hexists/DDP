#-*- coding: utf-8 -*-

import os, sys
import time
import datetime
import numpy as np
import tensorflow as tf
from optparse import OptionParser
##
from cnn import CNN
## import function
from data import load_xy, batch_iter
from common import get_tfconfig
from params import BATCH_SIZE

##
EMBEDDING_DIM = 128
FILTER_SIZES = "3,4,5"
NUM_FILTERS = 128
KEEP_PROB = 0.5
L2_REG_LAMBDA = 0.0
## 
NUM_CHECKPOINTS = 5
EVAL_EVERY = 100
CHECKPOINT_EVERY = 100
NUM_EPOCHS = 200
## https://github.com/dennybritz/cnn-text-classification-tf/blob/master/train.py
def preprocess():
    # load data
    x_text, y, max_document_length = load_xy()
    # 모든 문서에 등장하는 단어들에 인덱스를 할당하고, 길이가 다른 문서를 max_document_length로 맞춰주는 역할
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # randomly shuffle
    np.random.seed(10)
    ## arange : numpy 버전의 range , arange(start, < end, step)
    ## random.permutation : 순서를 무작위로 바꿈
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_suffled = x[shuffle_indices]
    y_suffled = y[shuffle_indices]
    # split train/test set
    dev_percentage = 0.1
    dev_idx = -1 * int(dev_percentage * float(len(y)))
    x_train, x_dev = x_suffled[:dev_idx], x_suffled[dev_idx:]
    y_train, y_dev = y_suffled[:dev_idx], y_suffled[dev_idx:]
    # debugging print 
    print("vocab size : {}".format(len(vocab_processor.vocabulary_)))
    print("train/dev split : {}/{}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    config = get_tfconfig()
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            with sess.as_default():
                cnn = CNN(x_train.shape[1], y_train.shape[1], len(vocab_processor.vocabulary_), EMBEDDING_DIM, list(map(int, FILTER_SIZES.split(","))), NUM_FILTERS, L2_REG_LAMBDA)
                gstep = tf.Variable(0,  name="gstep", trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                grads_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_vars, global_step=gstep)
                # output dictionary for models
                time_stamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", time_stamp))
                print("writing to {}".format(out_dir))
                # gradients summaries
                grad_summaries = []
                for g, v in grads_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        grad_summaries.append(grad_hist_summary)
                        ## sparsity - 얘는 뭐하는 친구였을까요?
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)
                ## 
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
                ##
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
                ## 
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)
                vocab_processor.save(os.path.join(out_dir, "vocab"))
                sess.run(tf.global_variables_initializer())
                def train_step(x_bat, y_bat):
                    feed_dict = {cnn.x : x_bat, cnn.y : y_bat, cnn.keep_prob : KEEP_PROB}
                    _,step, summaries, loss, acc = sess.run([train_op, gstep, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}. loss {}, acc {}".format(time_str, step, loss, acc))
                    train_summary_writer.add_summary(summaries, step)
                def dev_step(x_bat, y_bat, writer=None):
                    feed_dict = {cnn.x : x_bat, cnn.y : y_bat, cnn.keep_prob : 1.0}
                    step, summaries, loss, acc = sess.run([gstep, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}. loss {}, acc {}".format(time_str, step, loss, acc))
                    if writer:
                        writer.add_summary(summaries, step)
                    
                data = list(zip(x_train, y_train))
                batches = batch_iter(data, BATCH_SIZE, NUM_EPOCHS)
                for batch in batches:
                    x_bat, y_bat = zip(*batch)
                    train_step(x_bat, y_bat)
                    cur_step = tf.train.global_step(sess, gstep)
                    if cur_step % EVAL_EVERY == 0:
                        print("\nEvaluation : ")
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        print("")
                    if cur_step % CHECKPOINT_EVERY == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                        print("Saved model checkpoint to {}\n".format(path))
    
def main():
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)
    
if __name__ == '__main__':
    main()
    #parser = OptionParser()
    #parser.add_option("-", "--learning", dest="train_path", help="train file path", metavar="train_path")
    #(options, args) = parser.parse_args()

