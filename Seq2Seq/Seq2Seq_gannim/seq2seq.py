import tensorflow as tf
import numpy as np

class SEQ2SEQ(object):
    @staticmethod
    def get_rnn_cell(cell_type, n_hidden):
        if cell_type == 'rnn':
            return tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        elif cell_type == 'gru':
            return tf.nn.rnn_cell.GRUCell(n_hidden)
        elif cell_type == 'lstm':
            return tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        elif cell_type == 'bi-lstm':
            return tf.nn.rnn_cell.BasicLSTMCell(n_hidden), tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        return None, None 
    ##
    def bahdanau_attention(self, cell_type, n_hidden, dec_hidden, enc_outputs):
        ## Bahdanau Attention
        with tf.variable_scope('attention'):
            if cell_type in ['rnn', 'gru']:
                query = dec_hidden # ( batch, n_hidden )
                value = enc_outputs # ( batch, seq, n_hidden )
            elif cell_type == 'lstm':
                query = dec_hidden.h
                value = enc_outputs
            elif cell_type == 'bi-lstm':
                query = dec_hidden.h ##
                self.fw_enc_output, self.bw_enc_output = enc_outputs
                value = tf.add(self.fw_enc_output, self.bw_enc_output)
            # query shape ( , hidden )
            query_exp = tf.expand_dims(query, 1) # ( batch, 1, hidden)
            value_w = tf.layers.dense(value, n_hidden, activation=None, reuse=tf.AUTO_REUSE, name='value_w') # (batch, seq, hidden)
            query_exp_w = tf.layers.dense(query_exp, n_hidden, activation=None, reuse=tf.AUTO_REUSE, name='query_exp_w') # ( batch, 1, hidden)
            activation = tf.nn.tanh(value_w + query_exp_w) 
            self.att_score = tf.layers.dense(activation, 1, reuse=tf.AUTO_REUSE, name='att_score') # (batch, seq, 1)
            self.att_weights = tf.nn.softmax(self.att_score) # (batch, seq, 1)
            self.context_vector = tf.reduce_sum(self.att_weights * value, axis=1) # ( batch, n_hidden ) 
        return self.context_vector
    # uc_data : dataHelper object
    # cell_type : rnn cell type , {rnn, gru, lstm, bi-lstm}
    # n_hidden : hidden size
    # n_class : output type lenght
    def __init__(self, uc_data, cell_type, n_hidden, n_class, attention=False):
        ## input params
        self.enc_inputs = tf.placeholder(tf.int64, [None, None], name='enc_inputs') # (batch, step)
        self.dec_inputs = tf.placeholder(tf.int64, [None, None], name='dec_inputs') # (batch, step)
        self.x_sequence_length = tf.placeholder(tf.int64, name="x_sequence_length") # batch sequence_length 
        self.y_sequence_length = tf.placeholder(tf.int64, name="y_sequence_length") # batch sequence_length 
        self.out_keep_prob = tf.placeholder(tf.float32, name="out_keep_prob") # dropout
        ## local embedding
        self.enc_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))
        self.dec_embeddings = tf.Variable(tf.random_normal([uc_data.tot_dic_len, n_hidden]))
        ## encoder
        with tf.variable_scope('encode'):
            self.enc_input_embeddings = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs) 
            if cell_type in ['rnn', 'gru', 'lstm']:
                self.enc_cell = self.get_rnn_cell(cell_type, n_hidden)
                self.enc_cell = tf.nn.rnn_cell.DropoutWrapper(self.enc_cell, output_keep_prob=self.out_keep_prob)
                self.enc_outputs, self.enc_hidden = tf.nn.dynamic_rnn(self.enc_cell, self.enc_input_embeddings, sequence_length=self.x_sequence_length, dtype=tf.float32)
            elif cell_type == 'bi-lstm':
                self.fw_enc_cell, self.bw_enc_cell = self.get_rnn_cell(cell_type, n_hidden)
                self.fw_enc_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_enc_cell, output_keep_prob=self.out_keep_prob)
                self.bw_enc_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_enc_cell, output_keep_prob=self.out_keep_prob)
                self.enc_outputs, self.enc_hidden = tf.nn.bidirectional_dynamic_rnn(self.fw_enc_cell, self.bw_enc_cell, self.enc_input_embeddings, self.x_sequence_length, dtype=tf.float32)
        def false():
            return False
        def true():
            return True
        self.end_symbol_idx = tf.convert_to_tensor(np.array([[2]]), dtype=tf.int64) # cond
        self.targets = tf.placeholder(tf.int64, [None, None], name='target') # (batch, step)
        self.t_sequence_length = tf.placeholder(tf.int64, name="t_sequence_length") # batch sequence_length 
        ## decoder
        with tf.variable_scope('decode'):
            self.dec_input_embeddings = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs) # (batch size, sequence length , hidden size)
            if cell_type in ['rnn', 'gru', 'lstm']:
                self.dec_cell = self.get_rnn_cell(cell_type, n_hidden)
                self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob=self.out_keep_prob)
                self.outputs, self.dec_states = tf.nn.dynamic_rnn(self.dec_cell, self.dec_input_embeddings, initial_state=self.enc_hidden, sequence_length=self.y_sequence_length, dtype=tf.float32)
            elif cell_type == 'bi-lstm':
                self.fw_enc_hidden, self.bw_enc_hidden = self.enc_hidden
                c = tf.concat([self.fw_enc_hidden.c, self.bw_enc_hidden.c], 1)
                h = tf.concat([self.fw_enc_hidden.h, self.bw_enc_hidden.h], 1)
                self.add_enc_hidden = tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)
                ##
                self.dec_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden*2)
                self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob=self.out_keep_prob)
                self.context_vector = self.bahdanau_attention(cell_type, n_hidden, self.add_enc_hidden, self.enc_outputs) # (batch, hidden)
                if attention is True:
                    self.t_dec_input_embeddings = tf.transpose(self.dec_input_embeddings, [1,0,2]) # (t, batch, hidden)
                    def dec_cond(i, y_length, dstate, ot):
                        # end symbol index = 2
                        #i = tf.Print(i, [i, uc_data.max_outputs_seq_length, tf.greater_equal(i, uc_data.max_outputs_seq_length), tf.case({tf.greater_equal(i, uc_data.max_outputs_seq_length): false}, default=true)], "i > ")
                        return tf.case({tf.greater_equal(i, uc_data.max_outputs_seq_length): false}, default=true)
                    def dec_body(i, y_length, before_state, output_tensor_t):
                        i_dec_input = tf.transpose(tf.gather_nd(self.t_dec_input_embeddings, [[i]]), [1,0,2]) # (batch, 1, hidden)
                        i_dec_input = tf.concat([tf.expand_dims(self.context_vector, 1), i_dec_input], axis=-1) 
                        i_dec_outputs, i_dec_states = tf.nn.dynamic_rnn(self.dec_cell, i_dec_input, initial_state=before_state, dtype=tf.float32)
                        self.context_vector = self.bahdanau_attention(cell_type, n_hidden, i_dec_states, self.enc_outputs)
                        # i_dec_outputs :: (batch, 1, hidden)
                        output_tensor_t = output_tensor_t.write(i, tf.reshape(i_dec_outputs, [-1, n_hidden*2]))
                        return i+1, y_length, i_dec_states, output_tensor_t
                    ##
                    self.fw_enc_hidden, self.bw_enc_hidden = self.enc_hidden
                    c = tf.concat([self.fw_enc_hidden.c, self.bw_enc_hidden.c], 1)
                    h = tf.concat([self.fw_enc_hidden.h, self.bw_enc_hidden.h], 1)
                    self.add_enc_hidden = tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)
                    self.outputs = tf.TensorArray(tf.float32, size = 0, dynamic_size=True) #uc_data.max_targets_seq_length)
                    _, _, _, self.outputs = tf.while_loop(
                        cond=dec_cond,
                        body=dec_body,
                        loop_vars=[tf.constant(0), self.y_sequence_length, self.add_enc_hidden, self.outputs])
                    self.outputs = self.outputs.stack() # ( seq, batch, hidden )
                    self.outputs = tf.transpose(self.outputs, [1,0,2]) # (batch, seq, hidden)
                else:
                    self.outputs, self.dec_states = tf.nn.dynamic_rnn(self.dec_cell, self.dec_input_embeddings, initial_state=self.add_enc_hidden, sequence_length=self.y_sequence_length, dtype=tf.float32)
            # (batch, seq, hidden) - dense -> (batch, n_class, hidden)
            self.logits = tf.layers.dense(self.outputs, n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_layer')
            # loss
            self.t_mask = tf.sequence_mask(self.t_sequence_length, tf.shape(self.targets)[1])
            with tf.variable_scope("loss"):
                self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
                self.losses = self.losses * tf.to_float(self.t_mask)
                self.loss = tf.reduce_mean(self.losses)
            # accuracy
            with tf.variable_scope("accuracy"):
                self.prediction = tf.argmax(self.logits, 2)
                self.prediction_mask = self.prediction * tf.to_int64(self.t_mask)
                self.correct_pred = tf.equal(self.prediction_mask, self.targets)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"), name="accuracy")
        ## inferance decoder
        def cond(i, pred, dstate, ot, es):
            # end symbol index = 2
            p = tf.reshape(pred, [])
            e = tf.reshape(es, [])
            return tf.case({tf.greater_equal(i, uc_data.max_targets_seq_length): false, tf.equal(p, e): false}, default=true)
        def body(i, dec_before_inputs, before_state, output_tensor_t, end_symbol):
            with tf.variable_scope('decode'):
                inf_dec_input_embeddings = tf.nn.embedding_lookup(self.dec_embeddings, dec_before_inputs) # batch, step, hidden
                inf_dec_input_embeddings = tf.concat([tf.expand_dims(self.context_vector, 1), inf_dec_input_embeddings], axis=-1)
                inf_outputs, inf_dec_states = tf.nn.dynamic_rnn(self.dec_cell, inf_dec_input_embeddings, initial_state=before_state, sequence_length=[1], dtype=tf.float32)
                logits = tf.layers.dense(inf_outputs, n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_layer')
                inf_pred = tf.argmax(logits, 2)
                output_tensor_t = output_tensor_t.write( i, inf_pred )
                self.context_vector = self.bahdanau_attention(cell_type, n_hidden, inf_dec_states, self.enc_outputs)
            return i+1, inf_pred, inf_dec_states, output_tensor_t, end_symbol
        ##  run inferance
        self.inf_dec_inputs = tf.placeholder(tf.int64, [None, None], name='inf_dec_inputs') # (batch, step)
        self.output_tensor_t = tf.TensorArray(tf.int64, size = 0, dynamic_size=True) #uc_data.max_targets_seq_length)
        if cell_type == 'bi-lstm':
            self.fw_enc_hidden, self.bw_enc_hidden = self.enc_hidden
            c = tf.concat([self.fw_enc_hidden.c, self.bw_enc_hidden.c], 1)
            h = tf.concat([self.fw_enc_hidden.h, self.bw_enc_hidden.h], 1)
            self.add_enc_hidden = tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)
            self.context_vector = self.bahdanau_attention(cell_type, n_hidden, self.add_enc_hidden, self.enc_outputs)
            _, _, _, self.output_tensor_t, _ = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[tf.constant(0), self.inf_dec_inputs, self.add_enc_hidden, self.output_tensor_t, self.end_symbol_idx])
        else:
            _, _, _, self.output_tensor_t, _ = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[tf.constant(0), self.inf_dec_inputs, self.enc_hidden, self.output_tensor_t, self.end_symbol_idx])
        self.inf_result = self.output_tensor_t.stack()
        self.inf_result = tf.reshape( self.inf_result, [-1] , name='inf_result') 
