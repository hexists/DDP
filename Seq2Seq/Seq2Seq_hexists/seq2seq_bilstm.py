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

            self.enc_input_embedding = tf.nn.embedding_lookup(params=self.enc_embedding, ids=self.enc_input, name='enc_input_embedding')

            self.enc_outputs, self.enc_states = tf.nn.bidirectional_dynamic_rnn(self.fw_enc_cell, self.bw_enc_cell, self.enc_input_embedding, sequence_length=self.enc_input_len, dtype=tf.float32)
       
        # 디코더 셀을 구성한다.
        with tf.variable_scope('decode'):
            self.dec_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden * 2, name='basic_lstm_cell')
            self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob=self.output_keep_prob)

            self.dec_input_embedding = tf.nn.embedding_lookup(params=self.dec_embedding, ids=self.dec_input, name='dec_input_embedding')

            self.fw_enc_hidden, self.bw_enc_hidden = self.enc_states
            self.bi_enc_states_c = tf.concat([self.fw_enc_hidden.c, self.bw_enc_hidden.c], 1)
            self.bi_enc_states_h = tf.concat([self.fw_enc_hidden.h, self.bw_enc_hidden.h], 1)
            self.bi_enc_states = tf.nn.rnn_cell.LSTMStateTuple(c=self.bi_enc_states_c, h=self.bi_enc_states_h)

            # for bilstm
            self.dec_outputs, self.dec_states = tf.nn.dynamic_rnn(self.dec_cell, self.dec_input_embedding,
                                                   initial_state=self.bi_enc_states,
                                                   sequence_length=self.dec_input_len,
                                                   dtype=tf.float32)
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

        self.dec_end_idx = tf.constant([[dec_end_idx]], tf.int32)

        def infer_cond(i, inp_dec_state, inp_pred_output, output_tensor_t):
            def ret_true():
                return True
            def ret_false():
                return False
            def is_less():
                return tf.cond(tf.less(i, self.out_max_len), ret_true, ret_false)

            return tf.cond(tf.reshape(tf.equal(self.dec_end_idx, inp_pred_output), []), ret_false, is_less)
        
        def infer_body(i, inp_dec_state, inp_pred_output, output_tensor_t):
            with tf.variable_scope('decode'): 
                dec_input_embedding = tf.nn.embedding_lookup(params=self.dec_embedding, ids=inp_pred_output, name='dec_input_embedding')
                dec_output, dec_state = tf.nn.dynamic_rnn(self.dec_cell, dec_input_embedding, 
                                                            initial_state=inp_dec_state, dtype=tf.float32)
                model = tf.layers.dense(dec_output, self.out_voc_size, activation=None, reuse=tf.AUTO_REUSE)
                prediction = tf.argmax(model, 2, output_type=tf.int32)

                pred_output = prediction

                output_tensor_t = output_tensor_t.write(i, prediction)

            i = tf.add(i, 1)
        
            return [i, dec_state, pred_output, output_tensor_t]

        self.output_tensor_t = tf.TensorArray(tf.int32, size = self.out_max_len, name='output_tensor_t')

        _, _, _, self.output_tensor_t = tf.while_loop(
            cond = infer_cond,
            body = infer_body,
            loop_vars = [tf.constant(0), self.bi_enc_states, self.dec_input, self.output_tensor_t]
            )

        self.inference = self.output_tensor_t.stack()
        self.inference = tf.reshape(self.inference, [-1])
