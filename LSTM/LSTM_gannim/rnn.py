import tensorflow as tf
import numpy as np

class RNN(object):
    """
    A RNN for text classification.
    """
    def __init__(self, batch_size, vocab_size, embedding_size, num_hidden, num_classes):
        self.input_x = tf.placeholder(tf.int32, [batch_size, None]) # bactch , sentence_length
        self.input_y = tf.placeholder(tf.int32, [batch_size, num_classes]) # batch, num classes
        self.sequence_length = tf.placeholder(tf.int32, [batch_size]) # batch , 1 
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W") 
            #W = tf.get_variable('embedding_matrix', [vocab_size, embedding_size])
            self.embedded_words = tf.nn.embedding_lookup(W, self.input_x) # [1, batch, sentence_length, hidden]
            #self.embedded_words = tf.reshape(self.embedded_words, [batch_size, -1]) # [batch, sentence_length, hidden]

        # rnn layer
        with tf.device('/cpu:0'), tf.name_scope("rnn"):
            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)

        # cal rnn layer
        with tf.name_scope("cal_lstm"):
            init_state = self.rnn_cell.zero_state(batch_size, tf.float32)
            self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_cell, self.embedded_words, sequence_length=self.sequence_length, initial_state=init_state, dtype=tf.float32)
            # outputs (5, 15, 300)
            # states (2, 5, 300)
            self.outputs = tf.nn.dropout(self.outputs, self.dropout_keep_prob)
            self.last_outputs = self.states.h

        ## Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_hidden, num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[num_classes], initializer=tf.constant_initializer(0.0))
            self.scores = tf.nn.xw_plus_b(self.last_outputs, W, b, name="scores") # matmul(output, w) + b // logits
            self.predictions = tf.argmax(self.scores, 1, name="predictions") # tf.nn.softmax(scores) 

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1)) 
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))
