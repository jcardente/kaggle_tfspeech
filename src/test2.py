from os import listdir
from os.path import join, isfile, isdir, basename, splitext
import sys
import argparse
import tensorflow as tf
import numpy as np
import time
import util
import models

batchSize = 64

targetWords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

PARAMS = {
    'learningRates': [0.001,0.0001],
    'numEpochs': [14,4],
    'batchSize': 512,    
    'sampRate': 16000,
    'numSamples': 16000,
    'trainLimitInput': None,
    'trainShuffleSize': 5000,
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


def datasetTestBuildDataset(audioPath):
    dataset = []
    testFiles = listdir(audioPath)
    for fname in testFiles:
        fpath = join(audioPath, fname)        
        if not(fname.endswith('.wav') and isfile(fpath)):
            continue
        data, samprate = util.readWavFile(fpath)
        if len(data) < samprate:
           pad = np.zeros((samprate,1))
           start = (len(pad)-len(data))//2
           pad[start:start+len(data)] = data
           data = pad

        if len(data) > samprate:
           start = (len(data) - samprate)//2
           data = data[start:start+samprate]

        dataset.append({'file':fpath, 'label': -1, 'data':data, 'samprate': samprate})
                
    return dataset

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',type=str, default='./data/train/audio',
                        dest='audioDir',
                        help='Directory containing audio files')
    parser.add_argument('-c', type=str,
                        dest='chkpointFile',
                        required = True,
                        help='Checkpoint filename')
    parser.add_argument('-o', type=str,
                        dest='outputPath',
                        required = True,
                        help='Path to store submission file')
    FLAGS, unparsed = parser.parse_known_args()


    print('Loading audio data...')
    audioPath = FLAGS.audioDir
    dataset = datasetTestBuildDataset(audioPath)

    ############################################################
    # labels
    labels = ['unknown','silence'] + targetWords
    noutputs = len(labels)
    
    # parse one audio file to get types and dimensions
    tmpfeatures = util.doMFCC(dataset[0]['data'], dataset[0]['samprate'], PARAMS)       
    nsteps  = tmpfeatures.shape[0]
    ninputs = tmpfeatures.shape[1]
    
    # build input pipeline using a generator
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # Store labels in graph for inference
        class_labels = tf.constant(labels, dtype=tf.string, name="class_labels")

        test_gen     = util.makeInputGenerator(dataset, False, None, PARAMS)
        test_data    = tf.data.Dataset.from_generator(test_gen,
                                                       (tf.string, tf.int32, tf.float32),
                                                       ([], [],[nsteps,ninputs]))
        test_data    = test_data.batch(PARAMS['batchSize'])
        
        iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
        iterator = tf.data.Iterator.from_string_handle(iterator_handle,test_data.output_types, test_data.output_shapes)
        batch_fnames, batch_labels, batch_data = iterator.get_next()

        test_iterator   = test_data.make_one_shot_iterator()
        
    # Build the model
    with tf.device("/gpu:0"):
        #logits = dynamicRNN(batch_data, noutputs, 100)
        #logits = models.staticRNN(batch_data, noutputs, 10)
        #logits      = models.staticLSTM(batch_data, noutputs, 50)
        logits      = models.staticGRUBlock(batch_data, noutputs, 50)        
        xentropy    = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=logits)
        loss        = tf.reduce_mean(xentropy, name = "loss")
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        optimizer   = tf.train.AdamOptimizer(learning_rate = learning_rate)
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
        
    # Restore model
    print('Restoring model..')
    chkpointFile = FLAGS.chkpointFile
    metaFile     = chkpointFile + '.meta'
    if not isfile(metaFile):
        print("Error, meta file doesnt exist {}".format(metaFile))
        sys.exit(1)

    test_results = []              
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,chkpointFile)
        test_handle     = sess.run(test_iterator.string_handle())              
        # Get the prediction handle
        print('Starting predictions...')
        count = 0
        while True:
              try:
                  pred, names = sess.run([predClasses, clipnames], feed_dict={iterator_handle: test_handle})
                  names = [basename(n.decode('utf-8')) for n in names]
                  pred  = [c.decode('utf-8') for c in pred]

                  test_results.extend([(n,p) for n,p in zip(names, pred)])
                  count += 1
                  if (count % 1000) == 0:
                     print("    {}  batches completed".format(count))
              except tf.errors.OutOfRangeError:
                  break

    subFile =  splitext(basename(chkpointFile))[0] + "_sub_" + time.strftime('%Y%m%d_%H%M%S') + '.csv'
    of = open(join(FLAGS.outputPath, subFile),'w')
    of.write("fname,label\n")
    for n,p in test_results:
        of.write("{},{}\n".format(n,p))
    of.close()
