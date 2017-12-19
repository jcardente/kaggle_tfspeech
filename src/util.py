import os
import re
import hashlib
from os import listdir
from os.path import join, isfile, isdir
import numpy as np
import wave
from scipy.signal import hanning
import matplotlib.pyplot as plt
import matplotlib.ticker


# NB - From Kaggle data description page
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1 # ~134M

def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.  We want
    to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that
    testing samples will accidentally be reused in training when long runs
    are restarted for example. To keep this stability, a hash of the
    filename is taken and used to determine which set it should belong
    to. This determination only depends on the name and the set
    proportions, so it won't change as other files are added.  It's also
    useful to associate particular files as related (for example words
    spoken by the same person), so anything after 'nohash' in a filename
    is ignored for set determination. This ensures that
    'bobby_nohash_0.wav' and 'bobby_nohash_1.wav' are always in the same
    set, for example.  

    Args: 
    filename: File path of the data sample. 
    validation_percentage: How much of the data set to use for validation. 
    testing_percentage: How much of the data set to use for testing.

    Returns: 
    String, one of 'training', 'validation', or 'testing'.
    """
    
    # We want to ignore anything after 'nohash' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of 
    # grouping wavs that are close variations of each other.
    base_name = os.path.basename(filename)    
    hash_name = re.sub(r'nohash.*$', '', base_name).encode('utf-8')

    # This looks a bit magical, but we need to decide whether this file should 
    # go into the training, testing, or validation sets, and we want to keep 
    # existing files in the same set even if more files are subsequently 
    # added. 
    # To do that, we need a stable way of deciding based on just the file name 
    # itself, so we do a hash of that and then use that to generate a 
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else: result = 'training'

    return result 


def splitTrainData(audioPath, validationPercentage):
    labels = [x for x in listdir(audioPath) if isdir(join(audioPath,x)) and x[0] != '_']
    datasets = {'training': [],
                'testing': [],
                'validation': []}
    for label in listdir(audioPath):
        if (not isdir(join(audioPath,label))) or label == '_background_noise_':
            continue

        labelPath = join(audioPath,label)
        for fname in listdir(join(audioPath,label)):
            fpath = join(labelPath, fname)
            if not isfile(fpath):
                continue

            # Train or Validation?
            setname = which_set(fname, validationPercentage, 0)
            datasets[setname].append((labels.index(label), fpath))
                
    return labels, datasets
    
def splitLabelsData(datasetTuples):
    labels = []
    filenames = []
    for dt in datasetTuples:
        labels.append(dt[0])
        filenames.append(dt[1])
    return labels, filenames


# Wav preprocessing code heavily inspired by
# https://github.com/le1ca/spectrogram

def readWavFile(fname):
    w = wave.open(fname, 'r')
    framerate = w.getframerate()
    sampwidth = w.getsampwidth()
    nchannels = w.getnchannels()
    sampdtype = np.int8 if sampwidth==1 else np.int16
    data = w.readframes(w.getnframes())
    data = np.fromstring(data, dtype = sampdtype)
    data = data.astype(float) / np.max(np.abs(data))
    data = np.reshape(data,(len(data) // nchannels, nchannels))
    w.close()
    return [data, framerate]

def overlappedWindowIter(data, windowSize, overlapRate):
    start   = 0
    stepsz  = windowSize//overlapRate
    weights = np.array(hanning(windowSize)).reshape(windowSize,1)
    while True:
            window = data[start:(start+windowSize),]
            if len(window) != windowSize:
                    return
            yield window * weights
            start += stepsz
                
def doFFT(data):
    mindB = np.power(10.0, -120/20)  # Lowest signal level in dB
    y = np.fft.rfft(data)
    y = y[:len(data)//2]
    y = np.absolute(y) * 2.0 / len(data)
    #y = y / np.power(2.0, 8*nsampwidth - 1)
    #y = y / np.sum(y)
    y = y / np.max(y)
    y = 20 * np.log10(y.clip(mindB)) # clip before log to avoid log10 0 errors
    return y

def calcSpectrogram(fname, windowSize, overlapRate):
   data, framerate = readWavFile(fname)
   Y = [doFFT(x) for x in overlappedWindowIter(data, windowSize, overlapRate)]
   return np.column_stack(Y), framerate


def plotSpectrogram(Y, framerate, framesPerWindow, overlapRate):
    f = np.arange(framesPerWindow/2, dtype=np.float) * framerate / framesPerWindow
    t = np.arange(0, Y.shape[1], dtype=np.float) * framesPerWindow / framerate / overlapRate
    # PLOT THE SPECTOGRAM
    ax = plt.subplot(111)
    plt.pcolormesh(t, f, Y, vmin=-120, vmax=0)
    plt.yscale('symlog', linthreshy=100, linscaley=0.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim(0, t[-1])
    plt.ylim(0, f[-1])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    cbar = plt.colorbar()
    cbar.set_label("Intensity (dB)")
    plt.show()
