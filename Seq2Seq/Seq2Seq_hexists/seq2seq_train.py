#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import os
import sys
import pickle


START_SYMBOL = '^'
END_SYMBOL = '$'
PAD_SYMBOL = '+'
UNK_SYMBOL = '?'


class SEQ2SEQ:
    def __init__(self, inp_voc_size, out_voc_size, out_max_len, n_hidden, dec_end_idx, embedding=False):
        # 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
        self.inp_voc_size = inp_voc_size
        self.out_voc_size = out_voc_size
        self.out_max_len = out_max_len
        self.n_hidden = n_hidden
        self.embedding_size = 128

        with tf.variable_scope('input'):
            if embedding is True:
                # [batch size, time steps]
                self.enc_input = tf.placeholder(tf.int32, [None, None], name='enc_input')
                self.dec_input = tf.placeholder(tf.int32, [None, None], name='dec_input')
                self.inf_dec_input = tf.placeholder(tf.int32, [None, None], name='inf_dec_input')
                self.enc_embedding = tf.Variable(tf.random_normal([self.inp_voc_size, self.embedding_size]), name='enc_embedding')
                self.dec_embedding = tf.Variable(tf.random_normal([self.out_voc_size, self.embedding_size]), name='dec_embedding')
                self.inf_dec_embedding = tf.Variable(tf.random_normal([self.out_voc_size, self.embedding_size]), name='inf_dec_embedding')
            else:
                # [batch size, time steps, input size]
                self.enc_input = tf.placeholder(tf.float32, [None, None, inp_voc_size], name='enc_input')
                self.dec_input = tf.placeholder(tf.float32, [None, None, out_voc_size], name='dec_input')
                self.inf_dec_input = tf.placeholder(tf.float32, [None, None, out_voc_size], name='inf_dec_input')

            self.enc_input_len = tf.placeholder(tf.int64, [None], name='enc_input_len')
            self.dec_input_len = tf.placeholder(tf.int64, [None], name='dec_input_len')
            self.tgt_input_len = tf.placeholder(tf.int64, [None], name='tgt_input_len')

            # [batch size, time steps]
            self.targets = tf.placeholder(tf.int64, [None, None], name='targets')
            self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')

        # 인코더 셀을 구성한다.
        with tf.variable_scope('encode'):
            self.enc_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            self.enc_cell = tf.nn.rnn_cell.DropoutWrapper(self.enc_cell, output_keep_prob=self.output_keep_prob)
            if embedding is True:
                self.enc_input_embedding = tf.nn.embedding_lookup(params=self.enc_embedding, ids=self.enc_input, name='enc_input_embedding')
                self.enc_outputs, self.enc_states = tf.nn.dynamic_rnn(self.enc_cell, self.enc_input_embedding, sequence_length=self.enc_input_len, dtype=tf.float32)
            else:
                self.enc_outputs, self.enc_states = tf.nn.dynamic_rnn(self.enc_cell, self.enc_input, sequence_length=self.enc_input_len, dtype=tf.float32)
       
        # 디코더 셀을 구성한다.
        with tf.variable_scope('decode'):
            self.dec_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob=self.output_keep_prob)
            # Seq2Seq 모델은 인코더 셀의 최종 상태값을
            # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
            if embedding is True:
                self.dec_input_embedding = tf.nn.embedding_lookup(params=self.dec_embedding, ids=self.dec_input, name='dec_input_embedding')
                self.dec_outputs, self.dec_states = tf.nn.dynamic_rnn(self.dec_cell, self.dec_input_embedding,
                                                       initial_state=self.enc_states,
                                                       sequence_length=self.dec_input_len,
                                                       dtype=tf.float32)
            else:
                self.dec_outputs, self.dec_states = tf.nn.dynamic_rnn(self.dec_cell, self.dec_input,
                                                       initial_state=self.enc_states,
                                                       sequence_length=self.dec_input_len,
                                                       dtype=tf.float32)
            # dense
            self.model = tf.layers.dense(self.dec_outputs, self.out_voc_size, activation=None, reuse=tf.AUTO_REUSE)

        # Loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model, labels=self.targets))
        tf.summary.scalar('loss', self.loss)
             
        # Accuracy
        self.correct_predictions = tf.equal(tf.argmax(self.model, 2), self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)

        self.dec_end_idx = tf.constant([[dec_end_idx]], tf.int32)

        def infer_cond(i, inp_dec_state, inp_pred_output, output_tensor_t):
            def ret_true():
                return True
            def ret_false():
                return False
            def is_less():
                return tf.cond(tf.less(i, self.out_max_len), ret_true, ret_false)

            if embedding is True:
                return tf.cond(tf.reshape(tf.equal(self.dec_end_idx, inp_pred_output), []), ret_false, is_less)
            else:
                return tf.cond(tf.reshape(tf.equal(self.dec_end_idx, tf.argmax(inp_pred_output, 2)), []), ret_false, is_less)
        
        def infer_body(i, inp_dec_state, inp_pred_output, output_tensor_t):
            with tf.variable_scope('decode'): 
                if embedding is True:
                    inf_dec_input_embedding = tf.nn.embedding_lookup(params=self.inf_dec_embedding, ids=inp_pred_output, name='inf_dec_input_embedding')
                    dec_output, dec_state = tf.nn.dynamic_rnn(self.dec_cell, inf_dec_input_embedding, 
                                                                initial_state=inp_dec_state, dtype=tf.float32)
                else:
                    dec_output, dec_state = tf.nn.dynamic_rnn(self.dec_cell, inp_pred_output, 
                                                                initial_state=inp_dec_state, dtype=tf.float32)
                model = tf.layers.dense(dec_output, self.out_voc_size, activation=None, reuse=tf.AUTO_REUSE)
                prediction = tf.argmax(model, 2, output_type=tf.int32)

                if embedding is True:
                    pred_output = prediction
                else:
                    pred_output = tf.one_hot(prediction, self.out_voc_size)

                output_tensor_t = output_tensor_t.write(i, prediction)
            i = tf.add(i, 1)
        
            return [i, dec_state, pred_output, output_tensor_t]

        self.output_tensor_t = tf.TensorArray(tf.int32, size = self.out_max_len, name='output_tensor_t')

        _, _, _, self.output_tensor_t = tf.while_loop(
            cond = infer_cond,
            body = infer_body,
            loop_vars = [tf.constant(0), self.enc_states, self.inf_dec_input, self.output_tensor_t]
            )

        self.inference = self.output_tensor_t.stack()
        self.inference = tf.reshape(self.inference, [-1])


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS


def print_shape(label, data):
    print('{} => {}'.format(label, np.shape(data)))


def load_preprocessed(path):
    with open(path, mode='rb') as in_file:
        return pickle.load(in_file)


def split_data(inp_ids, out_ids, tgt_ids, dev_per=0.1):
    len_inp_ids = len(inp_ids)
    np.random.seed(10)
    shuf_idxs = np.random.permutation(np.arange(len_inp_ids))
    inp_shuf = inp_ids[shuf_idxs]
    out_shuf = out_ids[shuf_idxs]
    tgt_shuf = tgt_ids[shuf_idxs]

    dev_idx = int(dev_per * len_inp_ids)
    t_inp_ids, v_inp_ids = inp_ids[:dev_idx], inp_ids[dev_idx:]
    t_out_ids, v_out_ids = out_ids[:dev_idx], out_ids[dev_idx:]
    t_tgt_ids, v_tgt_ids = tgt_ids[:dev_idx], tgt_ids[dev_idx:]

    train = np.stack((t_inp_ids, t_out_ids, t_tgt_ids), axis=1)
    valid = np.stack((v_inp_ids, v_out_ids, v_tgt_ids), axis=1)
    
    return train, valid


def batch_iter(data, data_size, batch_size, num_epochs, is_shuf=True):
    '''
    num_epochs * (data_size / batch_size) 만큼 generator 반환
    '''
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if is_shuf is True:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            suffled_data = data[shuffle_indices]
        else:
            suffled_data = data
        for bnum in range(num_batches_per_epoch):
            sidx = bnum * batch_size
            eidx = min((bnum + 1) * batch_size, data_size)
            yield suffled_data[sidx:eidx]


def pad_and_one_hot(batch, max_len, dic_len, pad_idx, only_pad=False):
    n_batches = []
    n_lengths = []
    for bat in batch:
        bat = np.pad(bat, (0, max_len - len(bat)), 'constant')
        n_lengths.append(len(bat))
        if only_pad is True:
            n_batches.append(bat)
        else:
            n_batches.append(np.eye(dic_len)[bat])
    return n_batches, n_lengths


def transliterate(word, embedding=False):
    transliterated = []
    word_ids = [inp_voc2idx[w] for w in word]
    word_len = [len(word_ids)]

    if embedding is True:
        pred = [[out_voc2idx[START_SYMBOL]]]
        result = sess.run(seq2seq.inference, feed_dict={seq2seq.enc_input: [word_ids], seq2seq.enc_input_len:word_len, seq2seq.inf_dec_input:pred, seq2seq.output_keep_prob:1.0})
    else:
        word_one_hot = [np.eye(inp_voc_size)[word_ids]]
        pred_one_hot = [[np.eye(out_voc_size)[out_voc2idx[START_SYMBOL]]]]

        result = sess.run(seq2seq.inference, feed_dict={seq2seq.enc_input: word_one_hot, seq2seq.enc_input_len:word_len, seq2seq.inf_dec_input:pred_one_hot, seq2seq.output_keep_prob:1.0})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [out_idx2voc[i] for i in result]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    try:
        end = decoded.index('$')
        ret = ''.join(decoded[:end])
    except:
        ret = ''.join(decoded)
    return ret


#########
# 옵션 설정
######
learning_rate = 0.01
n_hidden = 128
total_epoch = 100
batch_size = 64


## DATA LOAD
path = './pickle/seq2seq.pickle'
(inp_ids, out_ids, tgt_ids), (inp_max_len, out_max_len, tgt_max_len), (inp_voc2idx, out_voc2idx), (inp_idx2voc, out_idx2voc) = load_preprocessed(path)
inp_voc_size, out_voc_size = len(inp_voc2idx), len(out_voc2idx)
train_set, valid_set = split_data(inp_ids, out_ids, tgt_ids)
train_set_len, valid_set_len = len(train_set), len(valid_set)
batches = batch_iter(train_set, train_set_len, batch_size, total_epoch)

DEBUG_MODE = False
use_embedding = True

words = ['ichadwick', 'peoria', 'whitepine', 'pancake', 'balloon', 'solen', 'richard', 'hubbard', 'mattox', 'stendhal']

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        seq2seq = SEQ2SEQ(inp_voc_size=inp_voc_size, out_voc_size=out_voc_size, out_max_len=out_max_len, n_hidden=n_hidden, dec_end_idx=out_voc2idx[END_SYMBOL], embedding=use_embedding)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(seq2seq.loss)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        
        summary_merge = tf.summary.merge_all()
        
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        valid_summary_dir = os.path.join(out_dir, "summaries", "valid")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())  # tf.Variable 초기화

        pad_idx = inp_voc2idx[PAD_SYMBOL]

        t_losses, t_accs = [], []
        for epoch, batch in enumerate(batches):
            inp_bat, out_bat, tgt_bat = zip(*batch)
            if use_embedding is True:
                inp_bat_one_hot, inp_bat_len = pad_and_one_hot(inp_bat, inp_max_len, inp_voc_size, pad_idx, only_pad=True)
                out_bat_one_hot, out_bat_len = pad_and_one_hot(out_bat, out_max_len, out_voc_size, pad_idx, only_pad=True)
                tgt_bat_pad, tgt_bat_len = pad_and_one_hot(tgt_bat, tgt_max_len, out_voc_size, pad_idx, only_pad=True)
            else:
                inp_bat_one_hot, inp_bat_len = pad_and_one_hot(inp_bat, inp_max_len, inp_voc_size, pad_idx)
                out_bat_one_hot, out_bat_len = pad_and_one_hot(out_bat, out_max_len, out_voc_size, pad_idx)
                tgt_bat_pad, tgt_bat_len = pad_and_one_hot(tgt_bat, tgt_max_len, out_voc_size, pad_idx, only_pad=True)

            # print('epoch = {}, batch = {}'.format(epoch, len(batch)))
            # print('out_bat_one_hot = {}'.format(np.shape(out_bat_one_hot)))
            # print('inp_bat_one_hot = {}'.format(np.shape(inp_bat_one_hot)))
            # print('tgt_bat_pad = {}'.format(np.shape(tgt_bat_pad)))

            feed = {seq2seq.enc_input: inp_bat_one_hot,
                seq2seq.dec_input: out_bat_one_hot,
                seq2seq.targets: tgt_bat_pad,
                seq2seq.enc_input_len: inp_bat_len,
                seq2seq.dec_input_len: out_bat_len,
                seq2seq.tgt_input_len: tgt_bat_len,
                seq2seq.output_keep_prob:0.5}

            if DEBUG_MODE is True:
                feed_inp = {
                    'inp_bat': seq2seq.enc_input,
                    'out_bat': seq2seq.dec_input,
                    'tgt_bat': seq2seq.targets,
                    'model': seq2seq.model,
                    'loss': seq2seq.loss,
                }
                results = sess.run(feed_inp, feed_dict=feed)
                for k, v in results.items():
                    print('k = {}, v = {}'.format(k, np.shape(v)))
                    # print(v)
                print()
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

                    v_batches = batch_iter(valid_set, valid_set_len, batch_size, 1)
                    v_losses, v_accs = [], []
                    for v_epoch, v_batch in enumerate(v_batches):
                        inp_bat, out_bat, tgt_bat = zip(*v_batch)
                        if use_embedding is True:
                            inp_bat_one_hot, inp_bat_len = pad_and_one_hot(inp_bat, inp_max_len, inp_voc_size, pad_idx, only_pad=True)
                            out_bat_one_hot, out_bat_len = pad_and_one_hot(out_bat, out_max_len, out_voc_size, pad_idx, only_pad=True)
                        else:
                            inp_bat_one_hot, inp_bat_len = pad_and_one_hot(inp_bat, inp_max_len, inp_voc_size, pad_idx)
                            out_bat_one_hot, out_bat_len = pad_and_one_hot(out_bat, out_max_len, out_voc_size, pad_idx)
                        tgt_bat_pad, out_bat_len = pad_and_one_hot(tgt_bat, tgt_max_len, out_voc_size, pad_idx, only_pad=True)
                        # print('v_epoch = {}'.format(v_epoch))
                        # print('v_inp_bat_one_hot = {}'.format(np.shape(inp_bat_one_hot)))
                        # print('v_out_bat_one_hot = {}'.format(np.shape(out_bat_one_hot)))
                        # print('v_tgt_bat_pad = {}'.format(np.shape(tgt_bat_pad)))

                        feed = {seq2seq.enc_input: inp_bat_one_hot,
                            seq2seq.dec_input: out_bat_one_hot,
                            seq2seq.targets: tgt_bat_pad,
                            seq2seq.enc_input_len: inp_bat_len,
                            seq2seq.dec_input_len: out_bat_len,
                            seq2seq.tgt_input_len: tgt_bat_len,
                            seq2seq.output_keep_prob: 1}

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
                    for word in words:
                        print('{} -> {}'.format(word, transliterate(word, use_embedding)))
        
        
        print('최적화 완료!')

        print('\n=== 번역 테스트 ===')

        for word in words:
            print('{} -> {}'.format(word, transliterate(word, use_embedding)))
