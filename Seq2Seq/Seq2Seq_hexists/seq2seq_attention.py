#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import os
import sys
import pickle
from utils import *


class SEQ2SEQ:
    def __init__(self, inp_voc_size, out_voc_size, out_max_len, n_hidden, dec_end_idx):
        self.inp_voc_size = inp_voc_size
        self.out_voc_size = out_voc_size
        self.out_max_len = out_max_len
        self.n_hidden = n_hidden
        self.embedding_size = 128  # input embedding size

        with tf.variable_scope('input'):
            # [batch size, time steps]
            self.enc_input = tf.placeholder(tf.int32, [None, None], name='enc_input')
            self.dec_input = tf.placeholder(tf.int32, [None, None], name='dec_input')

            self.initializer = tf.contrib.layers.xavier_initializer()

            self.enc_embedding = tf.get_variable(name='enc_embedding',
                                                shape=[self.inp_voc_size, self.embedding_size],
                                                dtype=tf.float32,
                                                initializer=self.initializer,
                                                trainable=True)
            self.dec_embedding = tf.get_variable(name='dec_embedding',
                                                shape=[self.out_voc_size, self.embedding_size],
                                                dtype=tf.float32,
                                                initializer=self.initializer,
                                                trainable=True)

            self.enc_input_len = tf.placeholder(tf.int64, [None], name='enc_input_len')
            self.dec_input_len = tf.placeholder(tf.int64, [None], name='dec_input_len')
            self.tgt_input_len = tf.placeholder(tf.int64, [None], name='tgt_input_len')

            # [batch size, time steps]
            self.targets = tf.placeholder(tf.int64, [None, None], name='targets')
            self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')

        # 인코더 셀을 구성한다.
        with tf.variable_scope('encode'):
            self.fw_enc_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, name='basic_lstm_cell')
            self.fw_enc_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_enc_cell, output_keep_prob=self.output_keep_prob)

            self.bw_enc_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, name='basic_lstm_cell')
            self.bw_enc_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_enc_cell, output_keep_prob=self.output_keep_prob)

            self.enc_input_embeddings = tf.nn.embedding_lookup(params=self.enc_embedding, ids=self.enc_input, name='enc_input_embeddings')

            self.enc_outputs, self.enc_states = tf.nn.bidirectional_dynamic_rnn(self.fw_enc_cell, self.bw_enc_cell, self.enc_input_embeddings, sequence_length=self.enc_input_len, dtype=tf.float32)

        # 디코더 셀을 구성한다.
        with tf.variable_scope('decode'):
            self.dec_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden * 2, name='basic_lstm_cell')
            self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob=self.output_keep_prob)

            self.dec_input_embeddings = tf.nn.embedding_lookup(params=self.dec_embedding, ids=self.dec_input, name='dec_input_embeddings')

            # from encoder
            self.fw_enc_hidden, self.bw_enc_hidden = self.enc_states
            self.bi_enc_states_c = tf.concat([self.fw_enc_hidden.c, self.bw_enc_hidden.c], 1)
            self.bi_enc_states_h = tf.concat([self.fw_enc_hidden.h, self.bw_enc_hidden.h], 1)
            self.bi_enc_states = tf.nn.rnn_cell.LSTMStateTuple(c=self.bi_enc_states_c, h=self.bi_enc_states_h)

            self.fw_enc_outputs, self.bw_enc_outputs = self.enc_outputs
            self.fw_bw_enc_outputs = tf.concat([self.fw_enc_outputs, self.bw_enc_outputs], 2)

            # (time, batch, hidden)
            self.time_dec_input_embeddings = tf.transpose(self.dec_input_embeddings, [1, 0, 2])

            self.dec_end_idx = tf.constant([[dec_end_idx]], tf.int32)

            def dec_cond(i, time_dec_input_embeddings, before_states, fw_bw_enc_outputs, outputs):
                def ret_true():
                    return True
                def ret_false():
                    return False
                # def is_less():
                return tf.cond(tf.less(i, self.out_max_len), ret_true, ret_false)
                # return tf.cond(tf.reshape(tf.equal(self.dec_end_idx, dec_output), []), ret_false, is_less)

            def dec_body(i, time_dec_input_embeddings, before_states, fw_bw_enc_outputs, outputs):
                context_vector = self.bahdanau_attention(before_states.h, fw_bw_enc_outputs)
                # (batch, 1, hidden)
                dec_input = tf.transpose(tf.gather_nd(time_dec_input_embeddings, [[i]]), [1,0,2])
                dec_context = tf.expand_dims(context_vector, 1)
                dec_input = tf.concat([dec_context, dec_input], axis=-1) 
                dec_outputs, dec_states = tf.nn.dynamic_rnn(self.dec_cell, dec_input, initial_state=before_states, dtype=tf.float32)

                # dec_outputs = (batch, 1, hidden)
                outputs = outputs.write(i, tf.reshape(dec_outputs, [-1, n_hidden * 2]))
                return i + 1, time_dec_input_embeddings, dec_states, fw_bw_enc_outputs, outputs

            self.output_tensor_t = tf.TensorArray(tf.float32, size = self.out_max_len, name='output_tensor_t')

            _, _, self.dec_states, _, self.output_tensor_t = tf.while_loop(
                cond = dec_cond,
                body = dec_body,
                loop_vars = [tf.constant(0), self.time_dec_input_embeddings, self.bi_enc_states, self.fw_bw_enc_outputs, self.output_tensor_t]
                )

            self.dec_outputs = self.output_tensor_t.stack()
            # (time, batch, hidden) => (batch, time, hidden)
            self.dec_outputs = tf.transpose(self.dec_outputs, [1, 0, 2])

            # dense
            self.model = tf.layers.dense(self.dec_outputs, self.out_voc_size, activation=None, reuse=tf.AUTO_REUSE)

        with tf.variable_scope('loss_accuracy'):
            # Loss
            # 1) sequence_mask => T, F
            # 2) boolean_mask = sequence_mask * result

            self.seq_mask = tf.sequence_mask(self.tgt_input_len, maxlen=self.out_max_len)

            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model, labels=self.targets)
            self.masked_losses = tf.boolean_mask(self.losses, self.seq_mask)
            self.loss = tf.reduce_mean(self.masked_losses, name='loss')
            tf.summary.scalar('loss', self.loss)
             
            # Accuracy
            self.correct_predictions = tf.equal(tf.argmax(self.model, 2), self.targets)
            self.masked_correct_predictions = tf.boolean_mask(self.correct_predictions, self.seq_mask)
            self.accuracy = tf.reduce_mean(tf.cast(self.masked_correct_predictions, "float"), name="accuracy")
            tf.summary.scalar('accuracy', self.accuracy)


    def bahdanau_attention(self, query, values):
        # 어텐션 메커니즘 구현
        # https://www.tensorflow.org/alpha/tutorials/text/nmt_with_attention
        with tf.variable_scope('attention'):
            # hidden shape == (batch_size, hidden size)
            # hidden_with_time_axis shape == (batch_size, 1, hidden size)
            # we are doing this to perform addition to calculate the score
            self.hidden_with_time_axis = tf.expand_dims(query, 1)

            self.W1 = tf.layers.dense(values, self.embedding_size, activation=None, reuse=tf.AUTO_REUSE, name='W1')
            self.W2 = tf.layers.dense(self.hidden_with_time_axis, self.embedding_size, activation=None, reuse=tf.AUTO_REUSE, name='W2')
            self.act = tf.nn.tanh(self.W1 + self.W2)

            self.V = tf.layers.dense(self.act, 1, activation=None, reuse=tf.AUTO_REUSE, name='V')

            # score shape == (batch_size, max_length, hidden_size)
            self.attention_score = self.V

            # attention_weights shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            self.attention_weights = tf.nn.softmax(self.attention_score, axis=2)

            # context_vector shape after sum == (batch_size, hidden_size)
            self.context_vector = tf.reduce_sum(self.attention_weights * values, axis=1)

        return self.context_vector
