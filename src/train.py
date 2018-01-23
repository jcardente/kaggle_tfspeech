#------------------------------------------------------------
# train.py
#
# Training and validation for Kaggle Tensorflow speech competition

import argparse
import tensorflow as tf
import numpy as np
import time
from timeit import default_timer as timer
import os
import yaml
import util
import models

FLAGS = None
targetWords          = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',type=str, default='./data/train/audio',
                        dest='audioDir',
                        help='Directory containing audio files')
    parser.add_argument('-p', type=str, default='./params.yml',
                        dest='paramFile',
                        help='Parameter file')
    parser.add_argument('-c', type=str, default='./chkpoints',
                        dest='chkpointDir',
                        help='Directory to store checkpoints')
    parser.add_argument('--nocheck',
                        action='store_true',
                        default=False,
                        dest='noCheck',
                        help='Dont save a checkpoint')
    FLAGS, unparsed = parser.parse_known_args()

    PARAMS = yaml.load(open(FLAGS.paramFile,'r'))
    
    # labels
    labels = ['unknown','silence'] + targetWords
    noutputs = len(labels)
    
    # Build training data set
    audioPath = FLAGS.audioDir
    print('Indexing and loading audio data.....')
    trainIndex = util.dataTrainIndex(audioPath, targetWords, PARAMS) 
    trainData  = util.dataTrainBuild(trainIndex, labels, PARAMS) 
    util.dataTrainLoad(trainData, PARAMS)
    backgrounds = util.dataBackgroundLoad(audioPath, PARAMS)

    print('Created {} training examples (pre-augmentation'.format(len(trainData['training'])))
          
    # NB - parse one audio file to get types and dimensions. This allows changing
    #      the parameters without editing the model
    tmpfeatures = util.doMFCC(trainData['training'][0]['data'], PARAMS)
    nsteps  = tmpfeatures.shape[0]
    ninputs = tmpfeatures.shape[1]
    

    tf.reset_default_graph()
    isTraining = tf.placeholder(tf.bool, name='istraining')
    

    with tf.device("/gpu:0"):
        batch_data    = tf.placeholder(tf.float32, shape=[None,nsteps,ninputs], name='batch_data')
        batch_labels  = tf.placeholder(tf.int32, shape=[None], name='batch_labels')        
        logits        = models.conv2DRnn(batch_data, noutputs, 50, isTraining)        
        xentropy      = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=logits)
        loss          = tf.reduce_mean(xentropy, name = "loss")
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        optimizer     = tf.train.AdamOptimizer(learning_rate = learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = optimizer.minimize(loss)
            
        class_probs = tf.nn.softmax(logits)
        
    with tf.device("/cpu:0"):
        correct    = tf.nn.in_top_k(logits, batch_labels, 1)
        accuracy   = tf.reduce_mean(tf.cast(correct, tf.float32))
        prediction = tf.argmax(class_probs,1, name = "prediction")
        
    saver       = tf.train.Saver()    
    init_op     = tf.global_variables_initializer()
    batchCount  = 0
    batchReportInterval = 10
    epochLearningRate   = 0.001
    trainTimeStart      = timer()
    with tf.Session() as sess:
        sess.run(init_op)

        # Training loop
        for epoch in range(sum(PARAMS['numEpochs'])):            
            print("Epoch " + str(epoch))

            tmpCount = 0
            for i in range(len(PARAMS['numEpochs'])):
                tmpCount += PARAMS['numEpochs'][i]
                if epoch < tmpCount:
                    epochLearningRate = PARAMS['learningRates'][i]
                    break
            
            timeStart = timer()
            for batch in util.inputGenerator(trainData['training'],True, backgrounds, PARAMS):                
                _ , batch_loss, batch_accuracy = sess.run([training_op, loss, accuracy], feed_dict={batch_data: batch['features'], batch_labels: batch['labels'], learning_rate: epochLearningRate, isTraining: 1})
                batchCount += 1                
                if batchCount % batchReportInterval == 0:
                    timeEnd = timer()
                    trainRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                    print("Batch {} loss {} accuracy {} rate {}".format(batchCount, batch_loss, batch_accuracy, trainRate))
                    timeStart = timer()

        trainTimeEnd = timer()
        print("Total Training Time {:.2f}m".format((trainTimeEnd-trainTimeStart)/60))
        
        if not FLAGS.noCheck:
            chkpointName = 'model_' + time.strftime('%Y%m%d_%H%M%S') + '.ckpt'
            chkpointFullName = os.path.join(FLAGS.chkpointDir, chkpointName)
            save_path = saver.save(sess, chkpointFullName)
                
        # Validation loop
        if PARAMS['validationPercentage'] > 0:
            print("Starting validation....")
            numCorrect   = 0
            numTotal     = 0
            checkCorrect = 0
            checkTotal   = 0
            batchCount   = 0
            valTimeStart = timer()            
            timeStart    = valTimeStart        
            for batch in util.inputGenerator(trainData['validation'],False, None, PARAMS):                
                batch_correct, batch_accuracy = sess.run([correct, accuracy], feed_dict={batch_labels: batch['labels'], batch_data: batch['features'], isTraining:0})
                numCorrect += np.sum(batch_correct)
                numTotal   += len(batch_correct)
                cumAccuracy = numCorrect / numTotal

                batchCount += 1            
                if batchCount % batchReportInterval == 0:
                    timeEnd = timer()
                    valRate = float(batchReportInterval* PARAMS['batchSize']) / (timeEnd - timeStart)
                    print("Batch {} Batch Accuracy {} Accum {} Rate {:.2f}".format(batchCount, batch_accuracy, cumAccuracy, valRate))
                    timeStart = timer()

            valTimeEnd = timer()
            print("Validation Correct: {}  Total: {} Accuracy {:.2f} Total Time {:.2f}m".format(numCorrect, numTotal, numCorrect/numTotal, (valTimeEnd-valTimeStart)/60))

            
        if not FLAGS.noCheck:
            print("Model saved to file {}".format(chkpointFullName))
        else:
            print("No checkpoint saved")




