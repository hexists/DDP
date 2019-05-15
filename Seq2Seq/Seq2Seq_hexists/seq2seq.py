#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import os
import sys

class SEQ2SEQ:
    def __init__(self, n_input, n_hidden, dec_end_idx):
        # 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
        self.n_class = self.n_input = self.dic_len = n_input
        self.n_hidden = n_hidden

        # [batch size, time steps, input size]
        self.enc_input = tf.placeholder(tf.float32, [None, None, n_input], name='enc_input')
        self.dec_input = tf.placeholder(tf.float32, [None, None, n_input], name='dec_input')
        self.inf_dec_input = tf.placeholder(tf.float32, [None, None, n_input], name='inf_dec_input')
        # [batch size, time steps]
        self.targets = tf.placeholder(tf.int64, [None, None], name='targets')
        self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')

        # 인코더 셀을 구성한다.
        with tf.variable_scope('encode'):
            self.enc_cell = tf.nn.rnn_cell.BasicRNNCell(self.n_hidden)
            self.enc_cell = tf.nn.rnn_cell.DropoutWrapper(self.enc_cell, output_keep_prob=self.output_keep_prob)
            self.enc_outputs, self.enc_states = tf.nn.dynamic_rnn(self.enc_cell, self.enc_input, dtype=tf.float32)
       
        # 디코더 셀을 구성한다.
        with tf.variable_scope('decode'):
            self.dec_cell = tf.nn.rnn_cell.BasicRNNCell(self.n_hidden)
            self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob=self.output_keep_prob)
            # Seq2Seq 모델은 인코더 셀의 최종 상태값을
            # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
            self.dec_outputs, self.dec_states = tf.nn.dynamic_rnn(self.dec_cell, self.dec_input,
                                                   initial_state=self.enc_states,
                                                   dtype=tf.float32)
            # dense
            self.model = tf.layers.dense(self.dec_outputs, self.n_class, activation=None, reuse=tf.AUTO_REUSE)

        # Loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model, labels=self.targets))
        tf.summary.scalar('loss', self.loss)
             
        # Accuracy
        self.correct_predictions = tf.equal(tf.argmax(self.model, 2), self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)

        self.dec_end_idx = tf.constant([[dec_end_idx]], tf.int64)

        def infer_cond(i, inp_dec_state, inp_pred_onehot_output, output_tensor_t):
            def ret_true():
                return True
            def ret_false():
                return False
            def is_less():
                return tf.cond(tf.less(i, self.dic_len), ret_true, ret_false)

            return tf.cond(tf.reshape(tf.equal(self.dec_end_idx, tf.argmax(inp_pred_onehot_output, 2)), []), ret_false, is_less)
        
        def infer_body(i, inp_dec_state, inp_pred_onehot_output, output_tensor_t):
            with tf.variable_scope('decode'): 
                dec_output, dec_state = tf.nn.dynamic_rnn(self.dec_cell, inp_pred_onehot_output, 
                                                            initial_state=inp_dec_state, dtype=tf.float32)
                model = tf.layers.dense(dec_output, self.n_class, activation=None, reuse=tf.AUTO_REUSE)
                prediction = tf.argmax(model, 2)
                pred_onehot_output = tf.one_hot(prediction, dic_len)
                output_tensor_t = output_tensor_t.write(i, prediction)
            i = tf.add(i, 1)
        
            return [i, dec_state, pred_onehot_output, output_tensor_t]

        self.output_tensor_t = tf.TensorArray(tf.int64, size = dic_len, name='output_tensor_t')

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

# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]

def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
        input = [num_dic[n] for n in seq[0]]
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        output = [num_dic[n] for n in ('S' + seq[1])]
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch


#########
# 옵션 설정
######
learning_rate = 0.01
n_hidden = 128
total_epoch = 100

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        seq2seq = SEQ2SEQ(n_input=dic_len, n_hidden=n_hidden, dec_end_idx=num_dic['E'])
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(seq2seq.loss)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        
        train_summary_merge = tf.summary.merge_all()
        
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        input_batch, output_batch, target_batch = make_batch(seq_data)

        sess.run(tf.global_variables_initializer())  # tf.Variable 초기화

        for epoch in range(total_epoch):
            _, train_loss, train_acc, train_summary, keep_prob = sess.run([optimizer, seq2seq.loss, seq2seq.accuracy, train_summary_merge, seq2seq.output_keep_prob],
                               feed_dict={seq2seq.enc_input: input_batch,
                                          seq2seq.dec_input: output_batch,
                                          seq2seq.targets: target_batch,
                                          seq2seq.output_keep_prob:0.5})
        
            print('Epoch:', '%04d' % (epoch + 1),
                  'loss =', '{:.6f}'.format(train_loss),
                  'acc =', '{:.6f}'.format(train_acc))
        
            train_summary_writer.add_summary(train_summary, epoch)
        
        print('최적화 완료!')

        def translate(word):
            seq_data = [word, 'P' * len(word)]
        
            translated = []
        
            input_batch, pred_onehot_batch, _ = make_batch([seq_data])
            st_pred_onehot_output = [[pred_onehot_batch[0][0]]]
        
            result = sess.run(seq2seq.inference, feed_dict={seq2seq.enc_input: input_batch, seq2seq.inf_dec_input:st_pred_onehot_output, seq2seq.output_keep_prob:1.0})
        
            # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
            decoded = [char_arr[i] for i in result]
        
            # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
            end = decoded.index('E')
            return ''.join(decoded[:end])


        print('\n=== 번역 테스트 ===')
        
        print('word ->', translate('word'))
        print('wodr ->', translate('wodr'))
        print('love ->', translate('love'))
        print('loev ->', translate('loev'))
        print('abcd ->', translate('abcd'))
