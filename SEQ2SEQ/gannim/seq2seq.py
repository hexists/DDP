#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from umcha_trans_data import UmChaTransData
"""
ref code 
https://github.com/beyondnlp/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)-Tensor.py
https://github.com/golbin/TensorFlow-Tutorials/blob/master/10%20-%20RNN/03%20-%20Seq2Seq.py
"""

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

##
enc_inputs = tf.placeholder(tf.float32, [None, None, n_class]) # (batch, step, class)
dec_inputs = tf.placeholder(tf.float32, [None, None, n_class]) # (batch, step, class)
targets = tf.placeholder(tf.int64, [None, None]) # (batch, step)
#targets = tf.placeholder(tf.int64, [None, None, n_class]) # (batch, step)

att = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
out = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))
out_keep_prob = tf.placeholder(tf.float32, name="out_keep_prob") # dropout
sequence_length = tf.placeholder(tf.int64, name="sequence_length") # batch sequence_length 
##
def one_hot(bats, max_len):
    new_bats = np.zeros((batch_size, max_len, n_class), dtype='float32')
    for bidx, bat in enumerate(bats):
        new_bats[bidx][:len(bat)] = [np.eye(n_class) [idx] for idx in bat]
    return new_bats

def get_att_weights(dec_output, enc_outputs):
    att_scores = []
    enc_outputs = tf.transpose(enc_outputs, [1, 0, 2]) # (step, batch, hidden)
    for i in range(n_step):
        score = tf.squeeze(tf.matmul(enc_outputs[i], att), 0) # (hidden)
        att_score = tf.tensordot(tf.squeeze(dec_output, [0, 1]), score, 1) # inner product make scalar value
        att_scores.append(att_score)
    return tf.reshape(tf.nn.softmax(att_scores), [1, 1, -1]) # [1, 1, step]
    
#model = []
#attention = []

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=out_keep_prob)
    enc_outputs, enc_hidden = tf.nn.dynamic_rnn(enc_cell, enc_inputs, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=out_keep_prob)
    ##
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_inputs, initial_state=enc_hidden, dtype=tf.float32)
    logits = tf.layers.dense(outputs, n_class, activation=None)
    # loss
    with tf.name_scope("loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        #losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        loss_mask = tf.sequence_mask(sequence_length, tf.shape(targets)[1])
        losses = losses * tf.to_float(loss_mask)
        loss = tf.reduce_mean(losses)
    # accuracy
    with tf.name_scope("accuracy"):
        prediction = tf.argmax(logits, 2)
        #answer = tf.argmax(targets, 2)
        #correct_pred = tf.equal(prediction, answer)#targets)
        correct_pred = tf.equal(prediction, targets)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Train and Test
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    ## train 
    batchs = uc_data.batch_iter(train_set, batch_size, epochs)
    avg_loss, avg_acc = 0, 0
    for epoch, batch in enumerate(batchs):
        x_bat, y_bat, t_bat = zip(*batch)
        x_bat, y_bat = one_hot(x_bat, uc_data.max_inputs_seq_length), one_hot(y_bat, uc_data.max_outputs_seq_length)
        nt_bat = np.zeros((batch_size, uc_data.max_outputs_seq_length))
        t_seq_len = np.zeros((batch_size))
        for idx in range(np.shape(t_bat)[0]):
            nt_bat[idx][:len(t_bat[idx])] = t_bat[idx]
            t_seq_len[idx] = len(t_bat[idx])
        #print "epoch >", epoch, np.shape(nt_bat)
        feed = {enc_inputs:x_bat, dec_inputs:y_bat, targets: nt_bat, out_keep_prob:keep_prob, sequence_length:t_seq_len}
        #feed = {enc_inputs:x_bat, dec_inputs:y_bat, targets: t_bat, out_keep_prob:keep_prob, sequence_length:t_seq_len}
        _, _loss, acc = sess.run([optimizer, loss, accuracy], feed_dict=feed)
        avg_loss += _loss
        avg_acc += acc
        if (epoch + 1) % 100 == 0:
            avg_loss /= 100
            avg_acc /= 100
            print('Epoch: %04d loss = %.6f / avg acc = %.6f ' % ((epoch + 1), avg_loss, avg_acc))
            avg_loss = 0
            avg_acc = 0

            dev_batchs = uc_data.batch_iter(dev_set, batch_size, 1)
            avg_dev_loss, avg_dev_acc = 0, 0
            for dev_ep, dev_batch in enumerate(dev_batchs):
                x_bat, y_bat, t_bat = zip(*batch)
                if dev_ep == 0:
                    print '{} ->'.format(''.join(uc_data.tot_idx_word_dic[n] for n in x_bat[0])),
                x_bat, y_bat = one_hot(x_bat, uc_data.max_inputs_seq_length), one_hot(y_bat, uc_data.max_outputs_seq_length)
                result = sess.run(prediction, feed_dict={enc_inputs:x_bat, dec_inputs: y_bat, out_keep_prob:1.0})
                if dev_ep == 0:
                    decoded_str = ''.join([uc_data.tot_idx_word_dic[n] for n in result[0]])
                    translated = ''.join(decoded_str[:decoded_str.index('E')])
                    print translated
                #if dev_ep == 100: 
                break # 뭔가 내맘같지 않아..
            #print(' dev > loss = %.6f / avg acc = %.6f ' % (avg_dev_loss/(dev_ep+1), avg_dev_acc/(dev_ep+1)))
    ##
