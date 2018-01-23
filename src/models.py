#------------------------------------------------------------
# models.py
#
# Model for Kaggle Tensorflow speech competition



def conv2DRnn(batch_data, noutputs, nhidden, isTraining):

    data  = tf.expand_dims(batch_data, -1)
    
    conv1 = tf.layers.conv2d(data, filters=64, kernel_size=[10,5], strides=[1,2],
                             padding='SAME', activation=None)
    conv1 = tf.layers.batch_normalization(conv1, training=isTraining, momentum=0.9)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1,3,5,1], strides=[1,1,2,1], padding="SAME")
    conv1 = tf.layers.dropout(conv1, rate=0.5, training=isTraining)
    
    conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=[10,5], strides=[1,2],
                             padding='SAME', activation=None)
    conv2 = tf.layers.batch_normalization(conv2, training=isTraining, momentum=0.9)
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")    
    conv2 = tf.layers.dropout(conv2, rate=0.5, training=isTraining)    
    
    conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=[10,5], strides=[1,1],
                             padding='SAME', activation=None)
    conv3 = tf.layers.batch_normalization(conv3, training=isTraining, momentum=0.9)
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1,3,5,1], strides=[1,1,1,1], padding="VALID")        
    conv3 = tf.layers.dropout(conv3, rate=0.5, training=isTraining)
    
    conv3shape = conv3.shape.as_list()
    squeezed   = tf.reshape(conv3, (-1, conv3shape[1], conv3shape[2]*conv3shape[3]))
    
    X_seqs     = tf.unstack(tf.transpose(squeezed, perm=[1,0,2]))
    basic_cell = tf.contrib.rnn.GRUBlockCellV2(num_units=128)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)

    fc1  = tf.layers.dense(states, 256, activation=None)
    fc1  = tf.layers.batch_normalization(fc1, training=isTraining, momentum=0.9)
    fc1  = tf.nn.relu(fc1)
    
    logits = tf.layers.dense(fc1, noutputs)
    
    return logits
