import tensorflow as tf
import numpy as np

'''
ref: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
'''

class TextCNN(object):

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Varible(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name='W')
            self.embedding_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                        self.embedding_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Maxpooling over the inputs
                pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool')
                pooled_outputs.append(pooled)
