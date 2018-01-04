#------------------------------------------------------------
# train.py
#
# Solution for the Kaggle TF Speech challenge
import argparse
import tensorflow as tf
import numpy as np
import time
from timeit import default_timer as timer

import util
import models

FLAGS = None

# DEFAULT PARAMETERS
# framesPerWindow      = 512
# overlapRate          = 4
# numSamples           = 16000
# validationPercentage = 5
# unknownPercentage    = 10
# silencePercentage    = 10
# numEpochs            = 8
# learningRate         = 0.001
# batchSize            = 64
targetWords          = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

PARAMS = {
    'numEpochs': 8,
    'learningRate': 0.001,
    'batchSize': 128,    
    'sampRate': 16000,
    'numSamples': 16000,
    'trainLimitInput': None,
    'trainShuffleSize': 3200,
    'validationPercentage': 5,
    'unknownPercentage': 10,
    'silencePercentage': 10,
    'silenceFileName':   '_silence_',
    'maxShiftSamps': int(16000/100),
    'backgroundLabel': '_background_noise_',
    'backgroundMinVol': 0.1,    
    'backgroundMaxVol': 0.5,
    'mfccWindowLen':  30.0/1000,
    'mfccWindowStride': 10.0/1000,     
    'mfccNumCep': 20
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',type=str, default='./data/train/audio',
                        dest='audioDir',
                        help='Directory containing audio files')
    FLAGS, unparsed = parser.parse_known_args()

    # labels
    labels = ['unknown','silence'] + targetWords
    noutputs = len(labels)
    
    # Build training data set
    audioPath = FLAGS.audioDir
    print('Indexing datasets.....')
    trainIndex = util.dataTrainIndex(audioPath, targetWords, PARAMS) 
    trainData  = util.dataTrainBuild(trainIndex, labels, PARAMS) 

    print('Loading audio data...')
    util.dataTrainLoad(trainData, PARAMS)
    backgrounds = util.dataBackgroundLoad(audioPath, PARAMS)
    
    # parse one audio file to get types and dimensions
    #tmpspectro, _ = util.calcSpectrogram(datasets['training'][0][1], framesPerWindow, overlapRate)
    #tmpspectro, _ = util.calcMFCC(datasets['training'][0]['file'])
    tmpfeatures = util.doMFCC(trainData['training'][0]['data'], trainData['training'][0]['samprate'], PARAMS)
    nsteps  = tmpfeatures.shape[0]
    ninputs = tmpfeatures.shape[1]
    
    # build input pipeline using a generator
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # Store labels in graph for inference
        class_labels = tf.constant(labels, dtype=tf.string, name="class_labels")
        
        # Training data set
        train_gen     = util.makeInputGenerator(trainData['training'], True, backgrounds, PARAMS)
        train_data    = tf.data.Dataset.from_generator(train_gen,
                                                       (tf.string, tf.int32, tf.float32),
                                                       ([],[],[nsteps,ninputs]))
        train_data    = train_data.shuffle(buffer_size=PARAMS['trainShuffleSize'])
        train_data    = train_data.batch(PARAMS['batchSize'])

        # Validation data set
        #val_gen     = makeInputGenerator(datasets['validation'])
        val_gen     = util.makeInputGenerator(trainData['validation'], False, None, PARAMS)
        val_data    = tf.data.Dataset.from_generator(val_gen,
                                                       (tf.string, tf.int32, tf.float32),
                                                       ([], [],[nsteps,ninputs]))
        val_data    = val_data.batch(PARAMS['batchSize'])

        iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
        iterator = tf.data.Iterator.from_string_handle(iterator_handle,train_data.output_types, train_data.output_shapes)
        batch_fnames, batch_labels, batch_data = iterator.get_next()

        train_iterator = train_data.make_initializable_iterator()
        val_iterator   = val_data.make_initializable_iterator()
        
    # Build the model
    #with tf.device("/gpu:0"):
    with tf.device("/cpu:0"):
        #logits = dynamicRNN(batch_data, noutputs, 100)
        #logits = models.staticRNN(batch_data, noutputs, 10)
        logits      = models.staticLSTM(batch_data, noutputs, 50)
        xentropy    = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=logits)
        loss        = tf.reduce_mean(xentropy, name = "loss")
        optimizer   = tf.train.AdamOptimizer(learning_rate = PARAMS['learningRate'])
        training_op = optimizer.minimize(loss)
        class_probs = tf.nn.softmax(logits)
    
    with tf.device("/cpu:0"):
        # Accuracy
        correct     = tf.nn.in_top_k(logits, batch_labels, 1)
        accuracy    = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Prediction
        prediction = tf.argmax(class_probs,1, name = "prediction")
        clipnames  = tf.identity(batch_fnames, name="clipnames")
        predClasses = tf.gather(class_labels, prediction, name="predicted_classes")

        
    # Start session for training and validation
    saver       = tf.train.Saver()    
    init_op     = tf.global_variables_initializer()
    losses      = []
    batchCount = 0
    batchReportInterval = 100
    with tf.Session() as sess:
        sess.run(init_op)

        # Training loop
        train_handle = sess.run(train_iterator.string_handle())
        for epoch in range(PARAMS['numEpochs']):            
            print("Epoch " + str(epoch))
            sess.run(train_iterator.initializer)
            timeStart = timer()
            while True:
                try:
                    _ , batch_loss, batch_accuracy, blabels, bpreds, bclasses = sess.run([training_op, loss, accuracy, batch_labels, prediction, predClasses], feed_dict={iterator_handle: train_handle})
                    losses.append(batch_loss)
                    if batchCount % batchReportInterval == 0:
                        timeEnd = timer()
                        batchRate = float(batchReportInterval) / (timeEnd - timeStart)
                        print("Batch {} loss {} accuracy {} rate {}".format(batchCount, batch_loss, batch_accuracy, batchRate))
                        timeStart = timer()
                    batchCount += 1
                except tf.errors.OutOfRangeError:
                    break
        ckptName = './chkpoints/model_' + time.strftime('%Y%m%d_%H%M%S') + '.ckpt'
        save_path = saver.save(sess, ckptName)
                
        # Validation loop
        print("Starting validation....")
        val_handle = sess.run(val_iterator.string_handle())
        sess.run(val_iterator.initializer)
        numCorrect   = 0
        numTotal     = 0
        checkCorrect = 0
        checkTotal   = 0
        batchCount  = 0
        while True:
            try:
                batch_correct, batch_accuracy, blabels, bpreds, bclasses = sess.run([correct, accuracy, batch_labels, prediction, predClasses], feed_dict={iterator_handle: val_handle})
                numCorrect += np.sum(batch_correct)
                numTotal   += len(batch_correct)
                cumAccuracy = numCorrect / numTotal
                  
                checkCorrect += np.sum(np.equal(blabels, bpreds))
                checkTotal += len(bpreds)
                checkAccuracy = checkCorrect/checkTotal                                
                if batchCount % 10 == 0:
                    print("Batch {} Batch Accuracy {} Accum {} Check {}".format(batchCount, batch_accuracy, cumAccuracy, checkAccuracy))
                batchCount += 1
            except tf.errors.OutOfRangeError:
                break

            
        print("Validation Correct: {}  Total: {} Accuracy {} Check: {}".format(numCorrect, numTotal, numCorrect/numTotal, checkCorrect/checkTotal))
        print("Model saved to file {}".format(ckptName))




