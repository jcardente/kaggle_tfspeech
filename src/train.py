#------------------------------------------------------------
# train.py
#
# Train a RNN solution for the Kaggle TF Speech challenge
import argparse
import util
import tensorflow as tf
import numpy as np
import time

FLAGS = None

# DEFAULT PARAMETERS
framesPerWindow      = 512
overlapRate          = 4
validationPercentage = 5
numEpochs             = 1
learningRate          = 0.001
batchSize             = 32

def makeInputGenerator(dataset, framesPerWindow, overlapRate):
    def gen():
        for elem in dataset:
            label = np.array(elem[0], dtype=np.int16)
            fname = elem[1]
            data, _ = util.calcSpectrogram(fname, framesPerWindow, overlapRate)
            yield label, np.transpose(data).astype(np.float32)
    return gen



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',type=str, default='./data/train/audio',
                        dest='audioDir',
                        help='Directory containing audio files')
    FLAGS, unparsed = parser.parse_known_args()

    # Build dataset of labels and filenames
    audioPath = FLAGS.audioDir
    labels, datasets = util.splitTrainData(audioPath, validationPercentage)
    noutputs = len(labels)
    
    # parse one audio file to get types and dimensions
    tmpspectro, _ = util.calcSpectrogram(datasets['training'][0][1], framesPerWindow, overlapRate)
    nsteps  = tmpspectro.shape[1]
    ninputs = tmpspectro.shape[0]
    
    # build input pipeline using a generator
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # Training data set
        train_gen     = makeInputGenerator(datasets['training'], framesPerWindow, overlapRate)
        train_data    = tf.data.Dataset.from_generator(train_gen,
                                                       (tf.int32, tf.float32),
                                                       ([],[nsteps,ninputs]))
        #train_data    = train_data.shuffle(buffer_size=1000)
        train_data    = train_data.batch(batchSize)


        # Validation data set
        val_gen     = makeInputGenerator(datasets['validation'], framesPerWindow, overlapRate)
        val_data    = tf.data.Dataset.from_generator(val_gen,
                                                       (tf.int32, tf.float32),
                                                       ([],[nsteps,ninputs]))
        val_data    = val_data.batch(batchSize)

        
        # Reinitializable iterator
        #iterator = train_data.make_initializable_iterator()
        iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        batch_labels, batch_data = iterator.get_next()

        train_init_op = iterator.make_initializer(train_data)
        val_init_op   = iterator.make_initializer(val_data)

        
    # Build the model
    with tf.device("/gpu:0"):
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=100)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, batch_data, dtype=tf.float32)
        logits = tf.layers.dense(states, noutputs)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=logits)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
        training_op = optimizer.minimize(loss)

    with tf.device("/cpu:0"):
        correct = tf.nn.in_top_k(logits, batch_labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Start the training loop
    saver = tf.train.Saver()    
    init_op = tf.global_variables_initializer()
    losses  = []
    batch_count = 0
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(train_init_op)
        for epoch in range(numEpochs):
            print("Epoch " + str(epoch))
            while True:
                try:
                    _ , batch_loss = sess.run([training_op, loss])
                    losses.append(batch_loss)
                    if batch_count % 100 == 0:
                        batch_accuracy = sess.run(accuracy)
                        print("Batch {} loss {} accuracy {}".format(batch_count, batch_loss, batch_accuracy))
                    batch_count += 1
                except tf.errors.OutOfRangeError:
                    break
        ckptName = './tmp/model_' + time.strftime('%Y%m%d_%H%M%S') + '.ckpt'
        print("Saving parameters to file {}".format(ckptName))
        save_path = saver.save(sess, ckptName)

                
        # Now do validation
        print("Starting validation....")
        sess.run(val_init_op)
        numCorrect = 0
        numTotal   = 0
        batch_count = 0
        while True:
            try:
                batch_correct, batch_accuracy = sess.run([correct, accuracy])
                numCorrect += np.sum(batch_correct)
                numTotal   += len(batch_correct)
                if batch_count % 100 == 0:
                    print("Batch {} accuracy {}".format(batch_count, batch_accuracy))
                batch_count += 1
            except tf.errors.OutOfRangeError:
                break

    print("Validation Correct: {}  Total: {}".format(numCorrect, numTotal))




