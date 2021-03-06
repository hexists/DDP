import tensorflow as tf
import numpy as np

class TextRNN(object):
    """
    A RNN for text classification.
    """
    def __init__(
      self, max_sequence_length, num_classes, vocab_size, embedding_size, num_hidden, batch_size):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, max_sequence_length], name="input_x")
        self.sequence_length = tf.placeholder(tf.int32, name="sequence_length")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_words = tf.nn.embedding_lookup(W, self.input_x) #[batch, n_timesteps, n_inputs]

        # rnn layer
        with tf.device('/cpu:0'), tf.name_scope("rnn"):
            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden)

        # cal rnn layer
        with tf.name_scope("cal_lstm"):
            self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_cell, self.embedded_words, sequence_length=self.sequence_length, dtype=tf.float32)
            self.last_outputs = self.states.h

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.last_outputs, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_hidden, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
