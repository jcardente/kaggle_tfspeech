from os import listdir
from os.path import join, isfile, isdir, basename, splitext
import sys
import argparse
import tensorflow as tf
import numpy as np
import time
import util

batchSize = 64

def datasetTestBuildDataset(audioPath):
    dataset = []
    testFiles = listdir(audioPath)
    for fname in testFiles[0:2000]:
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

    # Restore model
    print('Restoring model..')
    chkpointFile = FLAGS.chkpointFile
    metaFile     = chkpointFile + '.meta'
    if not isfile(metaFile):
        print("Error, meta file doesnt exist {}".format(metaFile))
        sys.exit(1)

    test_results = []              
    saver = tf.train.import_meta_graph(metaFile)
    with tf.Session() as sess:
        saver.restore(sess,chkpointFile)
        graph = tf.get_default_graph()
              
        # parse one audio file to get types and dimensions
        tmpfeatures = util.doMFCC(dataset[0]['data'], dataset[0]['samprate'])       
        nsteps  = tmpfeatures.shape[0]
        ninputs = tmpfeatures.shape[1]

        test_gen     = util.makeInputGenerator(dataset)
        test_data    = tf.data.Dataset.from_generator(test_gen,
                                                       (tf.string, tf.int32, tf.float32),
                                                       ([], [],[nsteps,ninputs]))
        test_data    = test_data.batch(batchSize)

              
        # Reinitializable iterator
        iterator_handle = graph.get_tensor_by_name('iterator_handle:0')              
        test_iterator   = test_data.make_one_shot_iterator()
        test_handle     = sess.run(test_iterator.string_handle())
              
        # Get the prediction handle
        pred_op   = graph.get_operation_by_name("predicted_classes")
        clipnames = graph.get_tensor_by_name("clipnames:0")
        #classLabels = graph.get_tensor_by_name("class_labels:0")
        print('Starting predictions...')
        count = 0
        while True:
              try:
                  pred, names = sess.run([pred_op.outputs[0], clipnames], feed_dict={iterator_handle: test_handle})
                  names = [basename(n.decode('utf-8')) for n in names]
                  pred  = [c.decode('utf-8') for c in pred]

                  test_results.extend([(n,p) for n,p in zip(names, pred)])
                  count += 1
                  if (count % 1000) == 0:
                     print("    {}  completed".format(count))
              except tf.errors.OutOfRangeError:
                  break

    subFile =  splitext(basename(chkpointFile))[0] + "_sub_" + time.strftime('%Y%m%d_%H%M%S') + '.csv'
    of = open(join(FLAGS.outputPath, subFile),'w')
    of.write("fname,label\n")
    for n,p in test_results:
        of.write("{},{}\n".format(n,p))
    of.close()
