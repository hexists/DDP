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
EVAL_EVERY = 100
NUM_CHECKPOINTS = 5
CHECKPOINT_EVERY = 100

uc_data = UmChaTransData()
sources_train, sources_dev, outputs_train, outputs_dev, targets_train, targets_dev = uc_data.get_suffled_data()
##
train_set = np.array([(x, outputs_train[idx], targets_train[idx]) for idx, x in enumerate(sources_train)])
dev_set = np.array([(x, outputs_dev[idx], targets_dev[idx]) for idx, x in enumerate(sources_dev)])

batch_size = 50
n_hidden = 128
epochs = 100
n_class = len(uc_data.tot_word_idx_dic)
##
learning_rate = 0.001
keep_prob = 0.5

############# graph ################
enc_inputs = tf.placeholder(tf.int64, [None, None]) # (batch, step)
dec_inputs = tf.placeholder(tf.int64, [None, None]) # (batch, step)
targets = tf.placeholder(tf.int64, [None, None]) # (batch, step)

out_keep_prob = tf.placeholder(tf.float32, name="out_keep_prob") # dropout
##
x_sequence_length = tf.placeholder(tf.int64, name="x_sequence_length") # batch sequence_length 
y_sequence_length = tf.placeholder(tf.int64, name="y_sequence_length") # batch sequence_length 
t_sequence_length = tf.placeholder(tf.int64, name="t_sequence_length") # batch sequence_length 

enc_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))
dec_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))

with tf.variable_scope('encode'):
    enc_input_embeddings = tf.nn.embedding_lookup(enc_embeddings, enc_inputs) 
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=out_keep_prob)
    enc_outputs, enc_hidden = tf.nn.dynamic_rnn(enc_cell, enc_input_embeddings, sequence_length=x_sequence_length, dtype=tf.float32)

with tf.variable_scope('decode'), tf.name_scope('decode'):
    dec_input_embeddings = tf.nn.embedding_lookup(dec_embeddings, dec_inputs) 
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=out_keep_prob)
    ##
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input_embeddings, initial_state=enc_hidden, sequence_length=y_sequence_length, dtype=tf.float32)
    logits = tf.layers.dense(outputs, n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_layer')
    # loss
    t_mask = tf.sequence_mask(t_sequence_length, tf.shape(targets)[1])
    with tf.name_scope("loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        losses = losses * tf.to_float(t_mask)
        loss = tf.reduce_mean(losses)
    # accuracy
    with tf.name_scope("accuracy"):
        prediction = tf.argmax(logits, 2)
        prediction_mask = prediction * tf.to_int64(t_mask)
        correct_pred = tf.equal(prediction_mask, targets)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")

## inferance
def cond(i, pred, dstate, ot):
    return tf.less(i, uc_data.max_targets_seq_length)
    
def body(i, dec_before_inputs, before_state, output_tensor_t):
    with tf.variable_scope('decode'), tf.name_scope('decode'):
        inf_dec_input_embeddings = tf.nn.embedding_lookup(dec_embeddings, dec_before_inputs) 
        inf_outputs, inf_dec_states = tf.nn.dynamic_rnn(dec_cell, inf_dec_input_embeddings, initial_state=before_state, dtype=tf.float32)
        logits = tf.layers.dense(inf_outputs, n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_layer')
        prediction = tf.argmax(logits, 2)
        output_tensor_t = output_tensor_t.write( i, prediction )
    return i+1, prediction, inf_dec_states, output_tensor_t

inf_dec_inputs = tf.placeholder(tf.int64, [None, None]) # (batch, step)
output_tensor_t = tf.TensorArray(tf.int64, size = uc_data.max_targets_seq_length)
_, _, _, output_tensor_t = tf.while_loop(
    cond=cond,
    body=body,
    loop_vars=[tf.constant(0), inf_dec_inputs, enc_hidden, output_tensor_t])
inf_result = output_tensor_t.stack()
inf_result = tf.reshape( inf_result, [-1] ) 

def get_umchar_str(result):
    decoded_str = ''.join([uc_data.tot_idx_word_dic[n] for n in result])
    if uc_data.END_SYMBOL in decoded_str:
        translated = ''.join(decoded_str[:decoded_str.index(uc_data.END_SYMBOL)])
    else:
        translated = decoded_str
    return translated

def get_tfconfig():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    return config 

def train():
    config = get_tfconfig()
    # Train and Test
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    with tf.Session(config=config) as sess:
        gstep = tf.Variable(0, name="gstep", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_vars, global_step=gstep)
    
        loss_summary = tf.summary.scalar("loss", loss)
        acc_summary = tf.summary.scalar("accuracy", accuracy)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_merge = tf.summary.merge_all()
    
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        ##
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        ##
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)
        uc_data.save_vocab(os.path.join(out_dir, "vocab"))
        ##
        init = tf.global_variables_initializer()
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

            feed = {enc_inputs:nx_bat, dec_inputs:ny_bat, targets: nt_bat, out_keep_prob:keep_prob, x_sequence_length:x_seq_len, y_sequence_length:y_seq_len, t_sequence_length:t_seq_len}
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
                step, summ, _loss, _, acc = sess.run([gstep, train_summary_merge, loss, train_op, accuracy ], feed_dict=feed)
                 
            avg_loss += _loss
            avg_acc += acc
            cur_step = tf.train.global_step(sess, gstep)
            train_summary_writer.add_summary(summ, step)
            if cur_step % EVAL_EVERY == 0:
                ## train avg loss , avg acc
                avg_loss /= EVAL_EVERY
                avg_acc /= EVAL_EVERY
                print('Epoch: %04d loss = %.6f / avg acc = %.6f' % (cur_step, avg_loss, avg_acc))
                avg_loss, avg_acc = 0, 0
                ##
                dev_batchs = uc_data.batch_iter(dev_set, batch_size, 1)
                avg_dev_loss, avg_dev_acc = 0, 0
                for dev_ep, dev_batch in enumerate(dev_batchs):
                    x_bat, y_bat, _ = zip(*batch)
                    if dev_ep == 0:
                        print 'devset {} ->'.format(''.join(uc_data.tot_idx_word_dic[n] for n in x_bat[0])),
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
                    result, dev_summ, dev_loss, dev_acc = sess.run([prediction, dev_summary_op, loss, accuracy], feed_dict={enc_inputs:ny_bat, dec_inputs: ny_bat, out_keep_prob:1.0, x_sequence_length:x_seq_len, y_sequence_length:y_seq_len, targets:nt_bat, t_sequence_length:t_seq_len})
                    avg_dev_loss += dev_loss
                    avg_dev_acc += dev_acc
                    if dev_ep == 0:
                        print get_umchar_str(result[0])
                blen = dev_ep+1
                dev_summary_writer.add_summary(dev_summ, cur_step)
                print("\nEvaluation : dev loss = %.6f / dev acc = %.6f" %(avg_dev_loss/blen, avg_dev_acc/blen))

                transiteration(sess, 'apple')
            if cur_step % CHECKPOINT_EVERY == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                print("Saved model checkpoint to {}\n".format(path))
        ## 학습 다하고 최종!
        print "final inferance"
        transiteration(sess, 'apple')
    
def transiteration(sess, input_word):
    x_bat = uc_data.get_input_idxs(input_word) # [[44 45 42]] 
    x_len = np.array([len(x_bat[0])]) # [[3]]
    y_bat = uc_data.get_input_idxs(uc_data.START_SYMBOL) #[[1]] 
    result = sess.run(inf_result, feed_dict={enc_inputs:x_bat, inf_dec_inputs:y_bat, x_sequence_length:x_len, out_keep_prob:1.0})
    print '{} -> {}'.format(input_word, get_umchar_str(result))
train()
