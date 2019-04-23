#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class CNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, embedding_matrix=None):
        self.x = tf.placeholder(tf.int32, [None, sequence_length], name="x")
        self.y = tf.placeholder(tf.int32, [None, num_classes], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout
        ## ??
        l2_loss  = tf.constant(0.0)
        with tf.device("/gpu:0"), tf.name_scope("embedding"):
            # [tokens, embedd size]
            self.embed_arr = np.array(embedding_matrix) #[vocab_size, embedding_dim]
            self.embed_init = tf.constant_initializer(self.embed_arr)
            self.embedding_weight = tf.get_variable(name="embedding_weight", initializer=self.embed_init, shape=self.embed_arr.shape, trainable=False)
            self.embedding_weight = tf.reshape(self.embedding_weight, [-1, embedding_size])
            ##
            #self.embedding_weight = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="embedding_weight")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding_weight, self.x)
            # [tokens, embedd size, 1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # num_filters = filter 개수
                # filter size = filter 사이즈
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, w, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # maxpooling 
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
                pooled_outputs.append(pooled)
        # combine
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # dropout 
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)
        with tf.name_scope("output"):
            w = tf.get_variable("w", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")
            self.predict = tf.argmax(self.scores, 1, name="predict")
        # loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        # accuracy
        with tf.name_scope("accuracy"):
            correct_pred= tf.equal(self.predict, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
