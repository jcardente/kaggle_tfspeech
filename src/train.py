#------------------------------------------------------------
# train.py
#
# Train a RNN solution for the Kaggle TF Speech challenge
import argparse
import tensorflow as tf
import numpy as np
import time

import util
import models

FLAGS = None

# DEFAULT PARAMETERS
framesPerWindow      = 512
overlapRate          = 4
validationPercentage = 5
numEpochs            = 5
learningRate         = 0.001
batchSize            = 64


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',type=str, default='./data/train/audio',
                        dest='audioDir',
                        help='Directory containing audio files')
    FLAGS, unparsed = parser.parse_known_args()

    # Build dataset of labels and filenames
    audioPath = FLAGS.audioDir
    print('Indexing datasets.....')
    labels, datasets = util.datasetBuildIndex(audioPath, validationPercentage)
    noutputs = len(labels)
    print('Loading audio data...')
    util.datasetLoadData(datasets)
    
    # parse one audio file to get types and dimensions
    #tmpspectro, _ = util.calcSpectrogram(datasets['training'][0][1], framesPerWindow, overlapRate)
    #tmpspectro, _ = util.calcMFCC(datasets['training'][0]['file'])
    tmpspectro = util.doMFCC(datasets['training'][0]['data'], datasets['training'][0]['samprate'])       
    nsteps  = tmpspectro.shape[0]
    ninputs = tmpspectro.shape[1]
    
    # build input pipeline using a generator
    tf.reset_default_graph()

    with tf.device("/cpu:0"):

        # Store labels in graph for inference
        class_labels = tf.constant(labels, dtype=tf.string, name="class_labels")
        
        # Training data set
        train_gen     = util.makeInputGenerator(datasets['training'])
        train_data    = tf.data.Dataset.from_generator(train_gen,
                                                       (tf.string, tf.int32, tf.float32),
                                                       ([],[],[nsteps,ninputs]))
        train_data    = train_data.shuffle(buffer_size=3200)
        train_data    = train_data.batch(batchSize)

        # Validation data set
        #val_gen     = makeInputGenerator(datasets['validation'])
        val_gen     = util.makeInputGenerator(datasets['validation'])
        val_data    = tf.data.Dataset.from_generator(val_gen,
                                                       (tf.string, tf.int32, tf.float32),
                                                       ([], [],[nsteps,ninputs]))
        val_data    = val_data.batch(batchSize)

        # Create feedable iterator
        #iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes, shared_name='input_iterator')
        #train_init_op = iterator.make_initializer(train_data)
        #val_init_op   = iterator.make_initializer(val_data)
        
        iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
        iterator = tf.data.Iterator.from_string_handle(iterator_handle,train_data.output_types, train_data.output_shapes)
        batch_fnames, batch_labels, batch_data = iterator.get_next()

        train_iterator = train_data.make_initializable_iterator()
        val_iterator   = val_data.make_initializable_iterator()
        
    # Build the model
    with tf.device("/gpu:0"):
        #logits = dynamicRNN(batch_data, noutputs, 100)
        logits = models.staticRNN(batch_data, noutputs, 10)
        #logits      = models.staticLSTM(batch_data, noutputs, 10)
        xentropy    = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=logits)
        loss        = tf.reduce_mean(xentropy, name = "loss")
        optimizer   = tf.train.AdamOptimizer(learning_rate = learningRate)
        training_op = optimizer.minimize(loss)
    
    with tf.device("/cpu:0"):
        # Accuracy
        correct     = tf.nn.in_top_k(logits, batch_labels, 1)
        accuracy    = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Prediction
        smax = tf.nn.softmax(logits)
        prediction = tf.argmax(smax,1, name = "prediction")
        clipnames  = tf.identity(batch_fnames, name="clipnames")
        predClasses = tf.gather(class_labels, prediction, name="predicted_classes")
        
    # Start the training loop
    saver       = tf.train.Saver()    
    init_op     = tf.global_variables_initializer()
    losses      = []
    batch_count = 0
    with tf.Session() as sess:
        sess.run(init_op)
        train_handle = sess.run(train_iterator.string_handle())
        for epoch in range(numEpochs):            
            print("Epoch " + str(epoch))
            sess.run(train_iterator.initializer)
            while True:
                try:
                    _ , batch_loss, batch_accuracy = sess.run([training_op, loss, accuracy], feed_dict={iterator_handle: train_handle})
                    losses.append(batch_loss)
                    if batch_count % 100 == 0:
                        print("Batch {} loss {} accuracy {}".format(batch_count, batch_loss, batch_accuracy))
                    batch_count += 1
                except tf.errors.OutOfRangeError:
                    break
        ckptName = './chkpoints/model_' + time.strftime('%Y%m%d_%H%M%S') + '.ckpt'
        print("Saving parameters to file {}".format(ckptName))
        save_path = saver.save(sess, ckptName)
                
        # Now do validation
        print("Starting validation....")
        #sess.run(val_init_op)
        val_handle = sess.run(val_iterator.string_handle())
        sess.run(val_iterator.initializer)
        numCorrect  = 0
        numTotal    = 0
        batch_count = 0
        while True:
            try:
                batch_correct, batch_accuracy = sess.run([correct, accuracy], feed_dict={iterator_handle: val_handle})
                numCorrect += np.sum(batch_correct)
                numTotal   += len(batch_correct)
                if batch_count % 10 == 0:
                    print("Batch {} accuracy {}".format(batch_count, batch_accuracy))
                batch_count += 1
            except tf.errors.OutOfRangeError:
                break

    print("Validation Correct: {}  Total: {}".format(numCorrect, numTotal))




