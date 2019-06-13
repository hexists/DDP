#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import os
import sys
import pickle
# from seq2seq_bilstm import *
from seq2seq_attention import *
from utils import *


params = {}


def transliterate(sess, seq2seq, word, data):
    inp_voc2idx, out_voc2idx, out_idx2voc = data['inp_voc2idx'], data['out_voc2idx'], data['out_idx2voc']

    transliterated = []
    word_ids = [inp_voc2idx[w] for w in word]
    word_len = [len(word_ids)]

    pred = [[out_voc2idx[START_SYMBOL]]]
    result = sess.run(seq2seq.inference, feed_dict={seq2seq.enc_input: [word_ids], seq2seq.enc_input_len:word_len, seq2seq.dec_input:pred, seq2seq.output_keep_prob:1.0})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [out_idx2voc[i] for i in result]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    try:
        end = decoded.index('$')
        ret = ''.join(decoded[:end])
    except:
        ret = ''.join(decoded)
    return ret


def load_data(FLAGS):
    ## DATA LOAD
    (inp_ids, out_ids, tgt_ids), (inp_max_len, out_max_len, tgt_max_len), (inp_voc2idx, out_voc2idx), (inp_idx2voc, out_idx2voc) = load_preprocessed(FLAGS.pickle_path)
    inp_voc_size, out_voc_size = len(inp_voc2idx), len(out_voc2idx)
    return {'inp_ids': inp_ids, 'out_ids': out_ids, 'tgt_ids': tgt_ids,
            'inp_max_len': inp_max_len, 'out_max_len': out_max_len, 'tgt_max_len': tgt_max_len,
            'inp_voc2idx': inp_voc2idx, 'out_voc2idx': out_voc2idx,
            'inp_idx2voc': inp_idx2voc, 'out_idx2voc': out_idx2voc,
            'inp_voc_size': inp_voc_size, 'out_voc_size': out_voc_size}
    

def train(FLAGS, data):
    inp_ids, out_ids, tgt_ids = data['inp_ids'], data['out_ids'], data['tgt_ids']
    inp_max_len, out_max_len, tgt_max_len = data['inp_max_len'], data['out_max_len'], data['tgt_max_len']
    inp_voc2idx, out_voc2idx = data['inp_voc2idx'], data['out_voc2idx']
    inp_idx2voc, out_idx2voc = data['inp_idx2voc'], data['out_idx2voc']
    inp_voc_size, out_voc_size = data['inp_voc_size'], data['out_voc_size']

    train_set, valid_set = split_data(inp_ids, out_ids, tgt_ids)

    train_set_len, valid_set_len = len(train_set), len(valid_set)

    batches = batch_iter(train_set, train_set_len, FLAGS.batch_size, FLAGS.num_epochs)
    
    words = ['ichadwick', 'peoria', 'whitepine', 'pancake', 'balloon', 'solen', 'richard', 'hubbard', 'mattox', 'stendhal']
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            seq2seq = SEQ2SEQ(inp_voc_size=inp_voc_size, out_voc_size=out_voc_size, out_max_len=out_max_len, n_hidden=FLAGS.embedding_dim, dec_end_idx=out_voc2idx[END_SYMBOL])
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(seq2seq.loss)
    
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            
            summary_merge = tf.summary.merge_all()
            
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    
            valid_summary_dir = os.path.join(out_dir, "summaries", "valid")
            valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)
    
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))  # tf.Variable 초기화
    
            pad_idx = inp_voc2idx[PAD_SYMBOL]
    
            t_losses, t_accs = [], []
            for epoch, batch in enumerate(batches):
                inp_bat, out_bat, tgt_bat = zip(*batch)
                inp_bat_one_hot, inp_bat_len = pad_and_one_hot(inp_bat, inp_max_len, inp_voc_size, pad_idx, only_pad=True)
                out_bat_one_hot, out_bat_len = pad_and_one_hot(out_bat, out_max_len, out_voc_size, pad_idx, only_pad=True)
                tgt_bat_pad, tgt_bat_len = pad_and_one_hot(tgt_bat, tgt_max_len, out_voc_size, pad_idx, only_pad=True)
    
                feed = {seq2seq.enc_input: inp_bat_one_hot,
                    seq2seq.dec_input: out_bat_one_hot,
                    seq2seq.targets: tgt_bat_pad,
                    seq2seq.enc_input_len: inp_bat_len,
                    seq2seq.dec_input_len: out_bat_len,
                    seq2seq.tgt_input_len: tgt_bat_len,
                    seq2seq.output_keep_prob: FLAGS.dropout_keep_prob}
    
                if FLAGS.debug_mode is True:
                    feed_inp = {
                        'inp_bat': seq2seq.enc_input,
                        'out_bat': seq2seq.dec_input,
                        'inp_bat_len': seq2seq.enc_input_len,
                        'out_bat_len': seq2seq.dec_input_len,
                        'tgt_bat': seq2seq.targets,
                        'model': seq2seq.model,
                        'enc_outputs': seq2seq.enc_outputs,
                        'enc_states': seq2seq.enc_states,
                        'bi_enc_states_c': seq2seq.bi_enc_states_c,
                        'bi_enc_states_h': seq2seq.bi_enc_states_h,
                        'bi_enc_states': seq2seq.bi_enc_states,
                        'fw_bw_enc_outputs': seq2seq.fw_bw_enc_outputs,
                        'dec_outputs': seq2seq.dec_outputs,
                        'dec_states': seq2seq.dec_states,
                        'seq_mask': seq2seq.seq_mask,
                        'losses': seq2seq.losses,
                        'masked_losses': seq2seq.masked_losses,
                        'loss': seq2seq.loss,
                        'correct_predictions': seq2seq.correct_predictions,
                        'masked_correct_predictions': seq2seq.masked_correct_predictions,
                        'accuracy': seq2seq.accuracy
                    }
                    results = sess.run(feed_inp, feed_dict=feed)
                    for k, v in results.items():
                        print('{}: {}'.format(k, np.shape(v)))
                        print(v)
                    print()
                    break
                else:
                    _, train_loss, train_acc, train_summary, keep_prob = sess.run([optimizer, seq2seq.loss, seq2seq.accuracy, summary_merge, seq2seq.output_keep_prob],
                                       feed_dict=feed)
            
                    train_summary_writer.add_summary(train_summary, epoch)
    
                    t_losses.append(train_loss)
                    t_accs.append(train_acc)
    
                    if epoch % 100 == 0:
                        train_avg_loss = np.mean(t_losses)
                        train_avg_acc = np.mean(t_accs)
                        t_losses, t_accs = [], []
                        print('Epoch:', '%04d' % (epoch + 1),
                              'loss =', '{:.6f}'.format(train_avg_loss),
                              'acc =', '{:.6f}'.format(train_avg_acc))
    
                        v_batches = batch_iter(valid_set, valid_set_len, FLAGS.batch_size, 1)
                        v_losses, v_accs = [], []
                        for v_epoch, v_batch in enumerate(v_batches):
                            inp_bat, out_bat, tgt_bat = zip(*v_batch)
                            inp_bat_one_hot, inp_bat_len = pad_and_one_hot(inp_bat, inp_max_len, inp_voc_size, pad_idx, only_pad=True)
                            out_bat_one_hot, out_bat_len = pad_and_one_hot(out_bat, out_max_len, out_voc_size, pad_idx, only_pad=True)
                            tgt_bat_pad, tgt_bat_len = pad_and_one_hot(tgt_bat, tgt_max_len, out_voc_size, pad_idx, only_pad=True)
    
                            feed = {seq2seq.enc_input: inp_bat_one_hot,
                                seq2seq.dec_input: out_bat_one_hot,
                                seq2seq.targets: tgt_bat_pad,
                                seq2seq.enc_input_len: inp_bat_len,
                                seq2seq.dec_input_len: out_bat_len,
                                seq2seq.tgt_input_len: tgt_bat_len,
                                seq2seq.output_keep_prob: 1.}
    
                            valid_loss, valid_acc, valid_summary, keep_prob = sess.run([seq2seq.loss, seq2seq.accuracy, summary_merge, seq2seq.output_keep_prob],
                                               feed_dict=feed)
                        
                            v_losses.append(valid_loss)
                            v_accs.append(valid_acc)
    
                        valid_avg_loss = np.mean(v_losses)
                        valid_avg_acc = np.mean(v_accs)
                        v_losses, v_accs = [], []
                        valid_summary_writer.add_summary(valid_summary, epoch)
    
                        print('Valid_Epoch:', '%04d' % (epoch + 1),
                              'loss =', '{:.6f}'.format(valid_avg_loss),
                              'acc =', '{:.6f}'.format(valid_avg_acc))

                        # for word in words:
                        #     print('{} -> {}'.format(word, transliterate(sess, seq2seq, word, data)))
            
            
            print('최적화 완료!')
    
            # print('\n=== 번역 테스트 ===')
            # for word in words:
            #     print('{} -> {}'.format(word, transliterate(sess, seq2seq, word, data)))


if __name__ == '__main__':
    # Data loading params
    tf.flags.DEFINE_float('dev_sample_percentage', .1, 'Percentage of the training data to use for validation')
    tf.flags.DEFINE_string('pickle_path', './pickle/seq2seq.pickle', 'Pickle File Path')
    
    # Model Hyperparameters
    tf.flags.DEFINE_string('word2vec', None, 'Word2vec file with pre-trained embeddings (default: None)')
    tf.flags.DEFINE_integer('embedding_dim', 256, 'Dimensionality of character embedding (default: 256)')
    tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default: 0.5)')
    tf.flags.DEFINE_float('learning_rate', 0.001, 'Learning Rate (default: 0.001)')
    
    # Training parameters
    tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size (default: 64)')
    tf.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs (default: 200)')
    tf.flags.DEFINE_integer('evaluate_every', 100, 'Evaluate model on dev set after this many steps (default: 100)')
    tf.flags.DEFINE_integer('checkpoint_every', 100, 'Save model after this many steps (default: 100)')
    tf.flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store (default: 5)')
    
    # Misc Parameters
    tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
    tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')
    tf.flags.DEFINE_boolean('debug_mode', False, 'Set debug mode(default: False)')
    
    FLAGS = tf.flags.FLAGS
    FLAGS(sys.argv)

    data = load_data(FLAGS)
    train(FLAGS, data)
