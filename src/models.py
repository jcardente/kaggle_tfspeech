import tensorflow as tf

def dynamicRNN(batch_data, noutputs, nhidden):
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=nhidden)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, batch_data, dtype=tf.float32)
    logits = tf.layers.dense(states, noutputs)
    return logits

def staticRNN(batch_data, noutputs, nhidden):
    X_seqs = tf.unstack(tf.transpose(batch_data, perm=[1,0,2]))
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=nhidden)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
    logits = tf.layers.dense(states, noutputs)
    return logits

def staticLSTM(batch_data, noutputs, nhidden):
    X_seqs = tf.unstack(tf.transpose(batch_data, perm=[1,0,2]))
    basic_cell = tf.contrib.rnn.LSTMCell(num_units=nhidden, use_peepholes=True)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
    flat_states = tf.stack(states, axis=1)
    flat_states = tf.reshape(flat_states, [-1,2*nhidden])
    logits = tf.layers.dense(flat_states, noutputs)
    return logits
