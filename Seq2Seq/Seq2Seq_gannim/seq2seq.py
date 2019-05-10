#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import os 
import time
reload(sys)
sys.setdefaultencoding('utf-8')
from umcha_trans_data import UmChaTransData
"""
ref code 
https://github.com/beyondnlp/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)-Tensor.py
https://github.com/golbin/TensorFlow-Tutorials/blob/master/10%20-%20RNN/03%20-%20Seq2Seq.py
"""

DEBUG=0

uc_data = UmChaTransData()
sources_train, sources_dev, outputs_train, outputs_dev, targets_train, targets_dev = uc_data.get_suffled_data()
##
train_set = np.array([(x, outputs_train[idx], targets_train[idx]) for idx, x in enumerate(sources_train)])
dev_set = np.array([(x, outputs_dev[idx], targets_dev[idx]) for idx, x in enumerate(sources_dev)])

batch_size = 50
n_hidden = 128
epochs = 10
n_class = len(uc_data.tot_word_idx_dic)
##
learning_rate = 0.001
keep_prob = 0.5

############# graph ################
enc_inputs = tf.placeholder(tf.int64, [None, None]) # (batch, step)
dec_inputs = tf.placeholder(tf.int64, [None, None]) # (batch, step)
targets = tf.placeholder(tf.int64, [None, None]) # (batch, step)
#targets = tf.placeholder(tf.int64, [None, None, n_class]) # (batch, step)

out_keep_prob = tf.placeholder(tf.float32, name="out_keep_prob") # dropout
##
x_sequence_length = tf.placeholder(tf.int64, name="x_sequence_length") # batch sequence_length 
y_sequence_length = tf.placeholder(tf.int64, name="y_sequence_length") # batch sequence_length 
t_sequence_length = tf.placeholder(tf.int64, name="t_sequence_length") # batch sequence_length 

att = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
out = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))

enc_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))
dec_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))

with tf.variable_scope('encode'):
    enc_input_embeddings = tf.nn.embedding_lookup(enc_embeddings, enc_inputs) 
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=out_keep_prob)
    enc_outputs, enc_hidden = tf.nn.dynamic_rnn(enc_cell, enc_input_embeddings, sequence_length=x_sequence_length, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_input_embeddings = tf.nn.embedding_lookup(dec_embeddings, dec_inputs) 
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=out_keep_prob)
    ##
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input_embeddings, initial_state=enc_hidden, sequence_length=y_sequence_length, dtype=tf.float32)
    logits = tf.layers.dense(outputs, n_class, activation=None)
    # loss
    t_mask = tf.sequence_mask(t_sequence_length, tf.shape(targets)[1])
    with tf.name_scope("loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        #losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        losses = losses * tf.to_float(t_mask)
        loss = tf.reduce_mean(losses)
    # accuracy
    with tf.name_scope("accuracy"):
        prediction = tf.argmax(logits, 2)
        prediction_mask = prediction * tf.to_int64(t_mask)
        #answer = tf.argmax(targets, 2)
        #correct_pred = tf.equal(prediction, answer)#targets)
        correct_pred = tf.equal(prediction_mask, targets)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Train and Test
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    train_summary_merge = tf.summary.merge_all()

    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    sess.run(init)
    ## train 
    batchs = uc_data.batch_iter(train_set, batch_size, epochs)
    avg_loss, avg_acc = 0, 0
    for epoch, batch in enumerate(batchs):
        x_bat, y_bat, t_bat = zip(*batch)
        nx_bat = np.zeros((batch_size, uc_data.max_inputs_seq_length)) 
        ny_bat = np.zeros((batch_size, uc_data.max_outputs_seq_length)) 
        nt_bat = np.zeros((batch_size, uc_data.max_targets_seq_length)) 
        x_seq_len = np.zeros((batch_size))
        y_seq_len = np.zeros((batch_size))
        t_seq_len = np.zeros((batch_size))
        ##
        for idx in range(np.shape(x_bat)[0]):
            nx_bat[idx][:len(x_bat[idx])] = x_bat[idx]
            x_seq_len[idx] = len(x_bat[idx])
        for idx in range(np.shape(y_bat)[0]):
            ny_bat[idx][:len(y_bat[idx])] = y_bat[idx]
            y_seq_len[idx] = len(y_bat[idx])
        for idx in range(np.shape(t_bat)[0]):
            nt_bat[idx][:len(t_bat[idx])] = t_bat[idx]
            t_seq_len[idx] = len(t_bat[idx])
        #print "epoch >", epoch, np.shape(nt_bat)
        feed = {enc_inputs:nx_bat, dec_inputs:ny_bat, targets: nt_bat, out_keep_prob:keep_prob, x_sequence_length:x_seq_len, y_sequence_length:y_seq_len, t_sequence_length:t_seq_len}
        #feed = {enc_inputs:x_bat, dec_inputs:y_bat, targets: t_bat, out_keep_prob:keep_prob, sequence_length:t_seq_len}

        if DEBUG :
            feed_opt={
                'loss':loss,
                'accuracy':accuracy,
                'prediction': prediction,
                'targets': targets,
                'logits': logits,
            }

            results = sess.run(feed_opt, feed_dict=feed)
            for key in results:
                val = results[key]
                print(key , ":", val, np.shape(val) )
                sys.exit(0)

        else:
            _loss, _, acc = sess.run([ loss, optimizer, accuracy ], feed_dict=feed)
             
        avg_loss += _loss
        avg_acc += acc
        if (epoch + 1) % 10 == 0:
            avg_loss /= 10
            avg_acc /= 10
            print('Epoch: %04d loss = %.6f / avg acc = %.6f (%.6f)' % ((epoch + 1), avg_loss, avg_acc, acc))
            avg_loss = 0
            avg_acc = 0

            dev_batchs = uc_data.batch_iter(dev_set, batch_size, 1)
            avg_dev_loss, avg_dev_acc = 0, 0
            for dev_ep, dev_batch in enumerate(dev_batchs):
                x_bat, y_bat, _ = zip(*batch)
                if dev_ep == 0:
                    print '{} ->'.format(''.join(uc_data.tot_idx_word_dic[n] for n in x_bat[0])),
                nx_bat = np.zeros((batch_size, uc_data.max_inputs_seq_length)) 
                ny_bat = np.zeros((batch_size, uc_data.max_outputs_seq_length)) 
                x_seq_len = np.zeros((batch_size))
                y_seq_len = np.zeros((batch_size))
                ##
                for idx in range(np.shape(x_bat)[0]):
                    nx_bat[idx][:len(x_bat[idx])] = x_bat[idx]
                    x_seq_len[idx] = len(x_bat[idx])
                for idx in range(np.shape(y_bat)[0]):
                    ny_bat[idx][:len(y_bat[idx])] = y_bat[idx]
                    y_seq_len[idx] = len(y_bat[idx])
                result = sess.run(prediction, feed_dict={enc_inputs:nx_bat, dec_inputs: ny_bat, out_keep_prob:1.0, x_sequence_length:x_seq_len, y_sequence_length:y_seq_len})
                if dev_ep == 0:
                    decoded_str = ''.join([uc_data.tot_idx_word_dic[n] for n in result[0]])
                    if '' in decoded_str:
                        translated = ''.join(decoded_str[:decoded_str.index('')])
                    else:
                        translated = decoded_str
                    print translated
                break # 뭔가 내맘같지 않아..
            #print(' dev > loss = %.6f / avg acc = %.6f ' % (avg_dev_loss/(dev_ep+1), avg_dev_acc/(dev_ep+1)))
