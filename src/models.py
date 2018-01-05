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
    fc1   =  tf.layers.dense(flat_states, nhidden)
    logits = tf.layers.dense(fc1, noutputs)
    return logits

def staticLSTMBlock(batch_data, noutputs, nhidden):
    X_seqs = tf.unstack(tf.transpose(batch_data, perm=[1,0,2]))
    basic_cell = tf.contrib.rnn.LSTMBlockCell(num_units=nhidden, use_peephole=True)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
    flat_states = tf.stack(states, axis=1)
    flat_states = tf.reshape(flat_states, [-1,2*nhidden])
    fc1   =  tf.layers.dense(flat_states, nhidden)
    logits = tf.layers.dense(fc1, noutputs)
    return logits

def staticGRUBlock(batch_data, noutputs, nhidden):
    X_seqs = tf.unstack(tf.transpose(batch_data, perm=[1,0,2]))
    basic_cell = tf.contrib.rnn.GRUBlockCellV2(num_units=nhidden)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
    #flat_states = tf.stack(states, axis=1)
    #flat_states = tf.reshape(flat_states, [-1,2*nhidden])
    fc1   =  tf.layers.dense(states, nhidden)
    logits = tf.layers.dense(fc1, noutputs)
    return logits


def staticGRUBlockDeep(batch_data, noutputs, nhidden):
    X_seqs = tf.unstack(tf.transpose(batch_data, perm=[1,0,2]))
    layers = [tf.contrib.rnn.GRUBlockCellV2(num_units=nhidden) for l in range(2)]
    multilayer_cell = tf.contrib.rnn.MultiRNNCell(layers) 
    output_seqs, states = tf.contrib.rnn.static_rnn(multilayer_cell, X_seqs, dtype=tf.float32)
    flat_states = tf.stack(states, axis=1)
    flat_states = tf.reshape(flat_states, [-1,2*nhidden])
    
    fc1   =  tf.layers.dense(flat_states, nhidden)
    logits = tf.layers.dense(fc1, noutputs)
    return logits


def convRnnHybrid(batch_data, noutputs, nhidden):

    data  = tf.expand_dims(batch_data, -1)
    conv1 = tf.layers.conv2d(data, filters=8, kernel_size=3, strides=[1,1], padding='VALID')
    conv2 = tf.layers.conv2d(conv1, filters=8, kernel_size=3, strides=[1,1], padding='VALID')
    conv3 = tf.layers.conv2d(conv2, filters=8, kernel_size=3, strides=[1,1], padding='VALID')
    conv4 = tf.layers.conv2d(conv3, filters=1, kernel_size=1, strides=[1,1], padding='VALID')     

    squeezed = tf.squeeze(conv4, axis=[-1])
    
    X_seqs = tf.unstack(tf.transpose(squeezed, perm=[1,0,2]))
    layers = [tf.contrib.rnn.GRUBlockCellV2(num_units=nhidden) for l in range(2)]
    multilayer_cell = tf.contrib.rnn.MultiRNNCell(layers) 
    output_seqs, states = tf.contrib.rnn.static_rnn(multilayer_cell, X_seqs, dtype=tf.float32)
    flat_states = tf.stack(states, axis=1)
    flat_states = tf.reshape(flat_states, [-1,2*nhidden])
    
    fc1   =  tf.layers.dense(flat_states, nhidden)
    logits = tf.layers.dense(fc1, noutputs)
    return logits
    
