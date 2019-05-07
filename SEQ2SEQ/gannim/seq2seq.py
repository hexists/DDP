#-*- coding: utf-8 -*-

# ref code https://github.com/beyondnlp/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)-Tensor.py

import tensorflow as tf
import numpy as np

sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

word_list = " ".join(sentences).split()
word_list = list(set(word_list)) # 중복 제거
word_dict = {w: i for i, w in enumerate(word_list)} # beer : 1
print word_dict
number_dict = {i: w for i, w in enumerate(word_list)} # 1 : beer
n_class = len(word_dict) # output class count

n_step = 5
n_hidden = 128

epochs = 2000

keep_prob = 0.5
learning_rate = 0.001

def make_batch(sentences):
    # n_class == 3
    # np.eye(n_class) == [[1, 0, 0] [0 1 0] [0 0 1]]
    #####
    # [[word_dict[n] for n in sentences[0].split()]] == [[1, 9, 10, 5, 6]]
    # [np.eye(n_class) [[1, 9, 10, 5, 6]]] == 
    # [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #  [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    #  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    #  [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]
    input_batch = [np.eye(n_class) [[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class) [[word_dict[n] for n in sentences[1].split()]]]
    # [[3, 8, 0, 4, 2]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]
    return input_batch, output_batch, target_batch

input_batch, output_batch, target_batch = make_batch(sentences)

enc_inputs = tf.placeholder(tf.float32, [None, None, n_class]) # (batch, step, class)
dec_inputs = tf.placeholder(tf.float32, [None, None, n_class]) # (batch, step, class)
targets = tf.placeholder(tf.int64, [1, n_step]) # (batch, step)

att = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
out = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))

model = []
attention = []

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=keep_prob)
    enc_outputs, enc_hidden = tf.nn.dynamic_rnn(enc_cell, enc_inputs, dtype=tf.float32)

def get_att_weights(dec_output, enc_outputs):
    att_scores = []
    enc_outputs = tf.transpose(enc_outputs, [1, 0, 2]) # (step, batch, hidden)
    for i in range(n_step):
        score = tf.squeeze(tf.matmul(enc_outputs[i], att), 0) # (hidden)
        att_score = tf.tensordot(tf.squeeze(dec_output, [0, 1]), score, 1) # inner product make scalar value
        att_scores.append(att_score)
    return tf.reshape(tf.nn.softmax(att_scores), [1, 1, -1]) # [1, 1, step]

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
    inputs = tf.transpose(dec_inputs, [1, 0, 2])
    hidden = enc_hidden
    for i in range(n_step):
        exp_inputs = tf.expand_dims(inputs[i], 1)
        dec_output, hidden = tf.nn.dynamic_rnn(dec_cell, exp_inputs, initial_state=hidden, dtype=tf.float32, time_major=True)
        att_weights = get_att_weights(dec_output, enc_outputs)
        attention.append(tf.squeeze(att_weights))
        context = tf.matmul(att_weights, enc_outputs)
        dec_output = tf.squeeze(dec_output, 0) # (1, step)
        context = tf.squeeze(context, 1) # (1, hidden)

        model.append(tf.matmul(tf.concat((dec_output, context), 1), out)) # (step, batch, class)

    trained_att = tf.stack([attention[0], attention[1], attention[2], attention[3], attention[4]], 0) # to show attention matrix
    model = tf.transpose(model, [1, 0, 2]) # (step, class)
    prediction = tf.argmax(model, 1)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Train and Test
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        input_batch, output_batch, target_batch = make_batch(sentences)
        feed = {enc_inputs:input_batch, dec_inputs:output_batch, targets: target_batch}
        _, loss, attention = sess.run([optimizer, cost, trained_att], feed_dict=feed)
        
        if (epoch + 1) % 400 == 0:
            print('Epoch: %04d cost = %.6f' % ((epoch + 1), loss))

    predict_batch = [np.eye(n_class) [[word_dict[n] for n in 'P P P P P'.split()]]]
    result = sess.run(prediction, feed_dict={enc_inputs:input_batch, dec_inputs: predict_batch})
    print sentences[0].split(), '->', [number_dict[n] for n in result[0]]

