#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import os
import sys
import pickle
from utils import *

'''
ref: https://www.tensorflow.org/beta/tutorials/text/transformer
'''

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0, name='look_ahead_mask')
    # (seq_len, seq_len)
    return mask


def create_padding_mask(seq, call_name):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32, name=call_name)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    # (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]    


def create_masks(inp, tar):
    with tf.variable_scope('create_masks'):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp, 'enc_padding_mask')

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp, 'dec_padding_mask')

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.

        # look_ahead_mask: (seq, seq)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        # dec_target_padding_mask: (batch, 1, 1, seq)
        dec_target_padding_mask = create_padding_mask(tar, 'dec_target_padding_mask')
        # combined_mask: (batch, 1, seq, seq)
        # value가 있는 곳만 0으로 되어 있음
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask, name='combined_mask')

        return enc_padding_mask, combined_mask, dec_padding_mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model, call_name):
    with tf.variable_scope(call_name):
        angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(params, x, call_name):
    with tf.variable_scope(call_name):
        d_model, dff = params['n_hidden'], params['dff']
        output = tf.layers.dense(x, dff, activation=tf.nn.relu)
        return tf.layers.dense(output, d_model)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
                    to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)    # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)    

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)    # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)    # (..., seq_len_q, depth_v)

    return output, attention_weights


def multi_head_attention(params, v, k, q, mask, call_name):

    with tf.variable_scope(call_name):
        def split_heads(x, batch_size, num_heads, depth):
            """Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
            """
            x = tf.reshape(x, (batch_size, -1, num_heads, depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        d_model, num_heads = params['n_hidden'], params['num_heads']

        depth = d_model // num_heads

        wq = tf.layers.dense(q, d_model)
        wk = tf.layers.dense(k, d_model)
        wv = tf.layers.dense(v, d_model)

        s_wq = split_heads(wq, tf.shape(wq)[0], num_heads, depth)
        s_wk = split_heads(wk, tf.shape(wk)[0], num_heads, depth)
        s_wv = split_heads(wv, tf.shape(wv)[0], num_heads, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(s_wq, s_wk, s_wv, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (tf.shape(scaled_attention)[0], -1, d_model))

        # (batch_size, seq_len_q, d_model)
        output = tf.layers.dense(concat_attention, d_model)

        return output, attention_weights


def encoder_layer(params, x, mask, call_name):
    with tf.variable_scope(call_name):
        num_heads, dff, dropout_rate, training = params['num_heads'], params['dff'], params['dropout_rate'], params['training']

        # (batch_size, input_seq_len, d_model)
        attn_output, _ = multi_head_attention(params, x, x, x, mask, '{}_mha'.format(call_name))
        attn_output = tf.layers.dropout(attn_output, dropout_rate, training=training, name='{}_dropout1'.format(call_name))
        # (batch_size, input_seq_len, d_model)
        out1 = tf.contrib.layers.layer_norm(x + attn_output, begin_norm_axis=-1, scope='{}_layer_norm1'.format(call_name))

        # (batch_size, input_seq_len, d_model)
        ffn_output = point_wise_feed_forward_network(params, out1, '{}_ffn'.format(call_name))
        ffn_output = tf.layers.dropout(ffn_output, dropout_rate, training=training, name='{}_dropout2'.format(call_name))

        # (batch_size, input_seq_len, d_model)
        out2 = tf.contrib.layers.layer_norm(out1 + ffn_output, begin_norm_axis=-1, scope='{}_layer_norm2'.format(call_name))

        return out2


def decoder_layer(params, x, enc_output, look_ahead_mask, padding_mask, call_name):
    with tf.variable_scope(call_name):
        dropout_rate, training = params['dropout_rate'], params['training']
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = multi_head_attention(params, x, x, x, look_ahead_mask, '{}_mha1'.format(call_name))
        attn1 = tf.layers.dropout(attn1, dropout_rate, training=training, name='{}_dropout1'.format(call_name))
        out1 = tf.contrib.layers.layer_norm(attn1 + x, begin_norm_axis=-1, scope='{}_layer_norm1'.format(call_name))

        # (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = multi_head_attention(params, enc_output, enc_output, out1, padding_mask, '{}_mha2'.format(call_name))
        attn2 = tf.layers.dropout(attn2, dropout_rate, training=training, name='{}_dropout2'.format(call_name))
        # (batch_size, target_seq_len, d_model)
        out2 = tf.contrib.layers.layer_norm(attn2 + out1, begin_norm_axis=-1, scope='{}_layer_norm2'.format(call_name))

        # (batch_size, target_seq_len, d_model)
        ffn_output = point_wise_feed_forward_network(params, out2, '{}_ffn'.format(call_name))
        ffn_output = tf.layers.dropout(ffn_output, dropout_rate, training=training, name='{}_dropout3'.format(call_name))
        # (batch_size, target_seq_len, d_model)
        out3 = tf.contrib.layers.layer_norm(ffn_output + out2, begin_norm_axis=-1, scope='{}_layer_norm3'.format(call_name))

        return out3, attn_weights_block1, attn_weights_block2


class Transformer:
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.inp_voc_size = params['inp_voc_size']
        self.out_voc_size = params['out_voc_size']
        self.inp_max_len = params['inp_max_len']
        self.out_max_len = params['out_max_len']
        self.d_model = params['n_hidden']  # input embedding size
        self.dff = params['dff'] = self.d_model * 4
        self.num_heads = params['num_heads']
        self.num_layers = params['num_layers']

        self.params = params

        assert self.d_model % self.num_heads == 0

        with tf.variable_scope('input'):
            # [batch size, time steps]
            self.enc_input = tf.placeholder(tf.int32, [None, None], name='enc_input')
            self.dec_input = tf.placeholder(tf.int32, [None, None], name='dec_input')

            self.initializer = tf.contrib.layers.xavier_initializer()

            self.enc_embedding = tf.get_variable(name='enc_embedding',
                                                shape=[self.inp_voc_size, self.d_model],
                                                dtype=tf.float32,
                                                initializer=self.initializer,
                                                trainable=True)
            self.dec_embedding = tf.get_variable(name='dec_embedding',
                                                shape=[self.out_voc_size, self.d_model],
                                                dtype=tf.float32,
                                                initializer=self.initializer,
                                                trainable=True)

            self.enc_input_len = tf.placeholder(tf.int64, [None], name='enc_input_len')
            self.dec_input_len = tf.placeholder(tf.int64, [None], name='dec_input_len')
            self.tgt_input_len = tf.placeholder(tf.int64, [None], name='tgt_input_len')

            # [batch size, time steps]
            self.target = tf.placeholder(tf.int64, [None, None], name='target')
            self.dropout_rate = self.params['dropout_rate'] = tf.placeholder(tf.float32, name='dropout_rate')
            self.training = self.params['training'] = tf.placeholder(tf.bool, name='training')

            self.enc_padding_mask, self.combined_mask, self.dec_padding_mask = create_masks(self.enc_input, self.dec_input)

        # Encoder
        with tf.variable_scope('encoder'):
            self.enc_input_embeddings = tf.nn.embedding_lookup(params=self.enc_embedding, ids=self.enc_input, name='enc_input_embeddings')
            # normalize
            self.enc_input_embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

            # get positional_encoding
            self.enc_pos_encoding = positional_encoding(self.inp_max_len, self.d_model, 'encoder_positional_encoding')[:, :tf.shape(self.enc_input_embeddings)[1], :]

            # add positional_encoding
            self.enc_input_embeddings += self.enc_pos_encoding

            # dropout
            self.enc_input_embeddings = tf.layers.dropout(self.enc_input_embeddings, self.dropout_rate, training=self.training, name='enc_input_dropout')

            # output
            self.enc_output = self.enc_input_embeddings
            self.enc_outputs = {}
            self.enc_mask = self.enc_padding_mask
            for i in range(self.num_layers):
                self.enc_output = encoder_layer(self.params, self.enc_output, self.enc_mask, 'encoder_layer_{}'.format(i))
                self.enc_outputs[i] = self.enc_output

        # Decoder
        with tf.variable_scope('decoder'):
            self.dec_input_embeddings = tf.nn.embedding_lookup(params=self.dec_embedding, ids=self.dec_input, name='dec_input_embeddings')
            # normalize
            self.dec_input_embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

            # get positional_encoding
            self.dec_pos_encoding = positional_encoding(self.out_max_len, self.d_model, 'decoder_positional_encoding')[:, :tf.shape(self.dec_input_embeddings)[1], :]
            # add positional_encoding
            self.dec_input_embeddings += self.dec_pos_encoding

            # dropout
            self.dec_input_embeddings = tf.layers.dropout(self.dec_input_embeddings, self.dropout_rate, training=self.training, name='dec_input_dropout')

            # output
            self.dec_output = self.dec_input_embeddings
            self.dec_outputs = {}
            self.dec_look_ahead_mask = self.combined_mask
            self.dec_padding_mask = self.dec_padding_mask
            self.attention_weights = {}
            self.dec_enc_output = self.enc_output
            for i in range(self.num_layers):
                self.dec_output, self.block1, self.block2 = decoder_layer(self.params, self.dec_output, self.dec_enc_output, self.dec_look_ahead_mask, self.dec_padding_mask, 'decoder_layer_{}'.format(i))
                self.dec_outputs[i] = self.dec_output
                self.attention_weights['decoder_layer{}_block1'.format(i + 1)] = self.block1
                self.attention_weights['decoder_layer{}_block2'.format(i + 1)] = self.block2

        # Final
        with tf.variable_scope('final'):
            self.model = tf.layers.dense(self.dec_output, self.out_voc_size, name='final')

        with tf.variable_scope('loss_accuracy'):
            # Loss
            # 1) sequence_mask => T, F
            # 2) boolean_mask = sequence_mask * result

            self.seq_mask = tf.sequence_mask(self.tgt_input_len, maxlen=self.out_max_len)

            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model, labels=self.target)
            self.masked_losses = tf.boolean_mask(self.losses, self.seq_mask)
            self.loss = tf.reduce_mean(self.masked_losses, name='loss')
            tf.summary.scalar('loss', self.loss)

            # Accuracy
            self.predictions = tf.argmax(self.model, 2)
            self.correct_predictions = tf.equal(self.predictions, self.target)
            self.masked_correct_predictions = tf.boolean_mask(self.correct_predictions, self.seq_mask)
            self.accuracy = tf.reduce_mean(tf.cast(self.masked_correct_predictions, "float"), name="accuracy")
            tf.summary.scalar('accuracy', self.accuracy)
