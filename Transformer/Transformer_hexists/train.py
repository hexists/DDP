#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import os
import sys
import pickle
from transformer import *
from utils import *


# np.set_printoptions(threshold=np.inf)


params = {}


def get_idx2voc(result, data, result_type='output'):
    inp_voc2idx, inp_idx2voc, out_voc2idx, out_idx2voc = data['inp_voc2idx'], data['inp_idx2voc'], data['out_voc2idx'], data['out_idx2voc']

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    if result_type == 'input':
        decoded = [inp_idx2voc[i] for i in result]
    else:
        decoded = [out_idx2voc[i] for i in result]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    try:
        end = decoded.index('$')
        ret = ''.join(decoded[:end])
    except:
        ret = ''.join(decoded)

    if ret == '':
        ret = 'None'

    return ret


def transliterate(sess, transformer, word, data):
    inp_voc2idx, out_voc2idx, out_idx2voc = data['inp_voc2idx'], data['out_voc2idx'], data['out_idx2voc']
    inp_max_len, out_max_len = data['inp_max_len'], data['out_max_len']
    inp_voc_size, out_voc_size = data['inp_voc_size'], data['out_voc_size']
    pad_idx = inp_voc2idx[PAD_SYMBOL]

    inp_bat = [[inp_voc2idx[w] for w in word]]
    inp_bat_pad, inp_bat_len = pad_and_one_hot(inp_bat, inp_max_len, inp_voc_size, pad_idx, only_pad=True)
    out_bat_pad = [[out_voc2idx[START_SYMBOL]]]
    out_bat_len = [len(out_bat_pad[0])]

    for i in range(out_max_len):
        # print('{}\tout_bat_pad = {}'.format(i, out_bat_pad))
        feed = {transformer.enc_input: inp_bat_pad,
                transformer.enc_input_len: inp_bat_len,
                transformer.dec_input: out_bat_pad,
                transformer.dec_input_len: out_bat_len,
                transformer.dropout_rate: 0.,
                transformer.training: False}
        infer_enc_output, infer_predictions = sess.run([transformer.enc_output, transformer.predictions], feed)
        infer_predicted_id = infer_predictions[0][-1]

        if infer_predicted_id == out_voc2idx[END_SYMBOL]:
            break

        # print('{}\tinfer_predictions = {}'.format(i+1, infer_predictions))
        # print('{}\tinfer_predicted_id = {}'.format(i+1, infer_predicted_id))
        out_bat_pad = np.concatenate([out_bat_pad, [[infer_predicted_id]]], axis=-1)
        out_bat_len = [len(out_bat_pad[0])]

    # infer_sample = get_idx2voc(infer_predictions[0], data)
    infer_out_bat = get_idx2voc(out_bat_pad[0], data)[1:]

    try:
        end = infer_out_bat.index('$')
        ret = ''.join(nfer_out_bat[:end])
    except:
        ret = ''.join(infer_out_bat)

    if ret == '':
        ret = 'None'
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
            params = {'batch_size': FLAGS.batch_size,
                      'inp_voc_size': inp_voc_size,
                      'out_voc_size': out_voc_size,
                      'inp_max_len': inp_max_len,
                      'out_max_len': out_max_len,
                      'n_hidden': FLAGS.embedding_dim,
                      'dec_end_idx': out_voc2idx[END_SYMBOL],
                      'num_layers': 1,
                      'num_heads': 8}
            transformer = Transformer(params)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(transformer.loss)
    
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
                inp_bat_pad, inp_bat_len = pad_and_one_hot(inp_bat, inp_max_len, inp_voc_size, pad_idx, only_pad=True)
                out_bat_pad, out_bat_len = pad_and_one_hot(out_bat, out_max_len, out_voc_size, pad_idx, only_pad=True)
                tgt_bat_pad, tgt_bat_len = pad_and_one_hot(tgt_bat, tgt_max_len, out_voc_size, pad_idx, only_pad=True)
    
                feed = {transformer.enc_input: inp_bat_pad,
                    transformer.dec_input: out_bat_pad,
                    transformer.target: tgt_bat_pad,
                    transformer.enc_input_len: inp_bat_len,
                    transformer.dec_input_len: out_bat_len,
                    transformer.tgt_input_len: tgt_bat_len,
                    transformer.dropout_rate: (1 - FLAGS.dropout_keep_prob),
                    transformer.training: True}
    
                if FLAGS.debug_mode is True:
                    feed_inp = {
                        'inp_bat': transformer.enc_input,
                        'out_bat': transformer.dec_input,
                        'inp_bat_len': transformer.enc_input_len,
                        'out_bat_len': transformer.dec_input_len,
                        'tgt_bat': transformer.target,
                        'enc_padding_mask': transformer.enc_padding_mask,
                        'combined_mask': transformer.combined_mask,
                        'dec_padding_mask': transformer.dec_padding_mask,
                        'model': transformer.model,
                        'enc_output': transformer.enc_output,
                        'dec_output': transformer.dec_output,
                        'seq_mask': transformer.seq_mask,
                        'losses': transformer.losses,
                        'masked_losses': transformer.masked_losses,
                        'loss': transformer.loss,
                        'correct_predictions': transformer.correct_predictions,
                        'masked_correct_predictions': transformer.masked_correct_predictions,
                        'accuracy': transformer.accuracy
                    }
                    results = sess.run(feed_inp, feed_dict=feed)
                    for k, v in results.items():
                        print('{}: {}'.format(k, np.shape(v)))
                        print(v)
                    print()
                    break
                else:
                    _, train_loss, train_acc, train_summary, dropout_rate, predictions = sess.run([optimizer, transformer.loss, transformer.accuracy, summary_merge, transformer.dropout_rate, transformer.predictions],
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
                            v_inp_bat, v_out_bat, v_tgt_bat = zip(*v_batch)
                            inp_bat_pad, inp_bat_len = pad_and_one_hot(v_inp_bat, inp_max_len, inp_voc_size, pad_idx, only_pad=True)
                            out_bat_pad, out_bat_len = pad_and_one_hot(v_out_bat, out_max_len, out_voc_size, pad_idx, only_pad=True)
                            tgt_bat_pad, tgt_bat_len = pad_and_one_hot(v_tgt_bat, tgt_max_len, out_voc_size, pad_idx, only_pad=True)

                            feed = {transformer.enc_input: inp_bat_pad,
                                transformer.dec_input: out_bat_pad,
                                transformer.target: tgt_bat_pad,
                                transformer.enc_input_len: inp_bat_len,
                                transformer.dec_input_len: out_bat_len,
                                transformer.tgt_input_len: tgt_bat_len,
                                transformer.dropout_rate: 0.,
                                transformer.training: False}
    
                            valid_loss, valid_acc, valid_summary, dropout_rate, valid_prediction = sess.run([transformer.loss, transformer.accuracy, summary_merge, transformer.dropout_rate, transformer.predictions],
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

                        for word in words:
                            print('{} -> {}'.format(word, transliterate(sess, transformer, word, data)))

            print('최적화 완료!')
    
            print('\n=== 번역 테스트 ===')
            for word in words:
                print('{} -> {}'.format(word, transliterate(sess, transformer, word, data)))


if __name__ == '__main__':
    # Data loading params
    tf.flags.DEFINE_float('dev_sample_percentage', .1, 'Percentage of the training data to use for validation')
    tf.flags.DEFINE_string('pickle_path', './pickle/transformer.pickle', 'Pickle File Path')
    
    # Model Hyperparameters
    tf.flags.DEFINE_string('word2vec', None, 'Word2vec file with pre-trained embeddings (default: None)')
    tf.flags.DEFINE_integer('embedding_dim', 256, 'Dimensionality of character embedding (default: 256)')
    tf.flags.DEFINE_float('dropout_keep_prob', 0.9, 'Dropout keep probability (default: 0.9)')
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
