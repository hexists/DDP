#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import os 
import time
reload(sys)
sys.setdefaultencoding('utf-8')
from data_helper import dataHelper
from seq2seq import SEQ2SEQ

DEBUG=0
EVAL_EVERY = 100
NUM_CHECKPOINTS = 5
CHECKPOINT_EVERY = 1000

uc_data = dataHelper()
cell_type = 'bi-lstm' #rnn'#'bi-lstm'
batch_size = 50
n_hidden = 128
epochs = 100
n_class = len(uc_data.tot_word_idx_dic)
##
learning_rate = 0.001
keep_prob = 0.5
seq2seq = SEQ2SEQ(uc_data, cell_type, n_hidden, n_class, True)
sources_train, sources_dev, outputs_train, outputs_dev, targets_train, targets_dev = uc_data.get_suffled_data()

train_set = np.array([(x, outputs_train[idx], targets_train[idx]) for idx, x in enumerate(sources_train)])
dev_set = np.array([(x, outputs_dev[idx], targets_dev[idx]) for idx, x in enumerate(sources_dev)])

def get_umchar_str(result):
    decoded_str = ''.join([uc_data.tot_idx_word_dic[n] for n in result])
    if uc_data.END_SYMBOL in decoded_str:
        translated = ''.join(decoded_str[:decoded_str.index(uc_data.END_SYMBOL)])
    else:
        translated = decoded_str
    return translated
    
def transiteration(sess, input_word):
    x_bat = uc_data.get_input_idxs(input_word) # [[44 45 42]] 
    x_len = np.array([len(x_bat[0])]) # [[3]]
    y_bat = uc_data.get_input_idxs(uc_data.START_SYMBOL) #[[1]] 
    result = sess.run(seq2seq.inf_result, feed_dict={seq2seq.enc_inputs:x_bat, seq2seq.inf_dec_inputs:y_bat, seq2seq.x_sequence_length:x_len, seq2seq.out_keep_prob:1.0})
    translated = get_umchar_str(result)
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
        grads_vars = optimizer.compute_gradients(seq2seq.loss)
        train_op = optimizer.apply_gradients(grads_vars, global_step=gstep)
    
        loss_summary = tf.summary.scalar("loss", seq2seq.loss)
        acc_summary = tf.summary.scalar("accuracy", seq2seq.accuracy)
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
            ## np.pad() 로 바꿔보기 ?!
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

            feed = {seq2seq.enc_inputs:nx_bat, seq2seq.dec_inputs:ny_bat, seq2seq.targets: nt_bat, seq2seq.out_keep_prob:keep_prob, seq2seq.x_sequence_length:x_seq_len, seq2seq.y_sequence_length:y_seq_len, seq2seq.t_sequence_length:t_seq_len}
            if DEBUG :
                feed_opt={
                    'loss':seq2seq.loss,
                    'accuracy':seq2seq.accuracy,
                    'prediction': seq2seq.prediction,
                    'targets': seq2seq.targets,
                    'logits': seq2seq.logits,
                }
                results = sess.run(feed_opt, feed_dict=feed)
                for key in results:
                    val = results[key]
                    print(key , ":", val, np.shape(val) )
                    sys.exit(0)
            else:
                step, summ, _loss, _, acc = sess.run([gstep, train_summary_merge, seq2seq.loss, train_op, seq2seq.accuracy ], feed_dict=feed)
                 
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
                    feed_opt = {
                        'prediction': seq2seq.prediction,
                        'dev_summary_op': dev_summary_op,
                        'loss': seq2seq.loss,
                        'accuracy': seq2seq.accuracy
                    }
                    feed = {
                        seq2seq.enc_inputs: nx_bat, 
                        seq2seq.dec_inputs: ny_bat, 
                        seq2seq.targets:nt_bat, 
                        seq2seq.x_sequence_length:x_seq_len, 
                        seq2seq.y_sequence_length:y_seq_len, 
                        seq2seq.t_sequence_length:t_seq_len,
                        seq2seq.out_keep_prob:1.0
                    }
                    results = sess.run(feed_opt, feed_dict=feed)
                    avg_dev_loss += results.get('loss')
                    avg_dev_acc += results.get('accuracy')
                    if dev_ep == 0:
                        input_word = get_umchar_str(x_bat[0])
                        print('inferance {} -> {}'.format(input_word, transiteration(sess, input_word)))
                        print('devset    {} -> {}'.format(input_word, get_umchar_str(results.get('prediction')[0])))
                blen = dev_ep+1
                dev_summary_writer.add_summary(results.get('dev_summary_op'), cur_step)
                print("\nEvaluation : dev loss = %.6f / dev acc = %.6f" %(avg_dev_loss/blen, avg_dev_acc/blen))
            if cur_step % CHECKPOINT_EVERY == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                print("Saved model checkpoint to {}\n".format(path) )
        #학습 다하고 최종!
        print("final inferance")
        input_word = 'apple'
        translated = transiteration(sess, input_word)
        print('{} -> {}'.format(input_word, translated))
train()
