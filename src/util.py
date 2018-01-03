import os
import re
import hashlib
from os import listdir
from os.path import join, isfile, isdir
import math
import random
import numpy as np
import wave
#from scipy.signal import hanning
from python_speech_features import mfcc
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


def dataTrainIndex(audioPath, targetWords, PARAMS):
    index = {'targets': {'training': [], 'validation': [], 'testing': []},
             'unknown': {'training': [], 'validation': [], 'testing': []}}
    for label in listdir(audioPath):
        if (not isdir(join(audioPath,label))):
            continue

        if label == '_background_noise_':
            continue
        
        labelPath = join(audioPath,label)
        for fname in listdir(join(audioPath,label)):
            fpath = join(labelPath, fname)
            if not(fname.endswith('.wav') and isfile(fpath)):
                continue

            # Train or Validation?
            setname  = which_set(fname, PARAMS['validationPercentage'], 0)
            typename = 'targets' if label in targetWords else 'unknown'
            index[typename][setname].append({'label': label, 'file':fpath})
                
    return index


def dataTrainBuild(index, labels, PARAMS) 
    trainData = {}
    l2i = {l:i for i,l in enumerate(labels)}
    for setname in index['targets'].keys():
        setsize = len(index['targets'][setname])
        unksize = int(math.ceil(setsize * PARAMS['unknownPercentage'] / 100))
        silsize = int(math.ceil(setsize * PARAMS['silencePercentage'] / 100))

        trainData[setname] = index['targets'][setname][:]
        for entry in trainData:
            entry['label'] = l2i[entry['label']]
        
        random.shuffle(index['unknown'][setname])
        unks = index['unknown'][setname][:unksize]
        unkLabel = l2i['unknown']
        for unk in unks:
            unk['label'] = unkLabel
        trainData[setname].extend(unks)

        silLabel = l2i['silence']
        sils = [{'label': silLabel, 'file': None}] * silsize
        trainData[setname].extend(sils)

        random.shuffle(trainData[setname])
        
    return trainData


def dataTrainLoad(trainData, PARAMS):
    numSamples = PARAMS['numSamples']
    for setname in trainData.keys():
        for entry in trainData[setname]:
            fname = entry['file']
            data  = []
            if fname:
                data, samprate = readWavFile(fname)
            else:
                data = np.zeros((numSamples,1))

            # NB - some audio files are not exactly one second long
            #      pad or truncate as necessary
            if len(data) < numSamples:
               pad = np.zeros((numSamples,1))
               start = (len(pad)-len(data))//2
               pad[start:start+len(data)] = data
               data = pad

            if len(data) > numSamples:
               start = (len(data) - numSamples)//2
               data = data[start:start+numSamples]

            entry['data']     = data
            entry['samprate'] = numSamples


def dataTrainShift(data, maxShiftSamps):
    shift = random.randint(-1*maxShiftSamps, maxShiftSamps)
    shifted = np.zeros(data.shape)
    if shift >= 0:
        shifted[shift:] = data[:(len(data)-shift)]
    else:
        shifted[:(len(data)-shift)] = data[shift:]
    return shifted
        
def dataBackgroundLoad(audioPath):
    backgrounds = []
    for fname in listdir(join(audioPath,'_background_')):
        fpath = join(labelPath, fname)
        if not(fname.endswith('.wav') and isfile(fpath)):
            continue
        data, samprate = readWavFile(fpath)
        backgrounds.append(data)
    return backgrounds
            
def dataBackgroundMixin(data, backgrounds, maxvol):
    if len(backgrounds) == 0:
        return data
    bg      = random.choice(backgrounds)
    bgstart = random.randrange(len(bg)-len(data))
    bgvol   = random.random() * maxvol
    mixed   = (1.0-bgvol)*data + bgvol*bg[bgstart:(bgstart+len(data))]
    return mixed
    

def makeInputGenerator(dataset, doAugment, backgrounds, PARAMS):
    def gen():
        for elem in dataset:
            label = np.array(elem['label'], dtype=np.int16)
            fname = elem['file'].encode('utf-8')
            data  = elem['data']
            samprate = elem['samprate']
            if doAugment:
                data = dataTrainShift(data, PARAMS['maxShiftSamps']))
                data  = dataBackgroundMixin(data, backgrounds, PARAMS['maxBackgroundvol'])
            
            features = doMFCC(elem['data'],elem['samprate'], PARAMS)
            yield fname, label, features.astype(np.float32)
    return gen



            
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
    if np.max(np.abs(data)) > 0:
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
    if np.max(y) > 0:
        y = y / np.max(y)
    y = 20 * np.log10(y.clip(mindB)) # clip before log to avoid log10 0 errors
    return y

def calcSpectrogram(fname, windowSize, overlapRate):
   data, framerate = readWavFile(fname)
   
   Y = [doFFT(x) for x in overlappedWindowIter(data, windowSize, overlapRate)]
   Y = np.column_stack(Y)

   # NB - some audio samples are shorter than 1 sec. pad with zeros   
   sampsExpected = 1 + (framerate - windowSize) // (windowSize // overlapRate)
   if Y.shape[1] < sampsExpected:
       padding = np.ones((Y.shape[0],sampsExpected)) * -120
       delta   = sampsExpected - Y.shape[1]
       leftpad = delta // 2
       padding[:,leftpad:leftpad+Y.shape[1]] = Y
       Y = padding
   
   assert not np.any(np.isnan(Y))
   return Y, framerate


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

    
def doMFCC(data, sampRate, PARAMS):
    winlen  = PARAMS['mfccWindowLen']
    winstep = PARAMS['mfccWindowStride']
    numcep  = PARAMS['mfccNumCep']    
    mfccCoefs = mfcc(data, sampRate, nfilt=2*numcep, winlen=winlen, winstep=winstep, numcep=numcep)

    # NB - don't return the first MFCC coefficient
    return mfccCoefs[:,1:]


def plotMFCC(data, winlen, winstep, numcep):
    t = np.arange(0,data.shape[0]) * winstep
    c = np.arange(0,data.shape[1])
    ax = plt.subplot(111)
    plt.pcolormesh(t,c, data.T)
    #plt.yscale('symlog', linthreshy=100, linscaley=0.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim(0, t[-1])
    plt.ylim(0, c[-1])
    #plt.xlabel("Time (s)")
    #plt.ylabel("Coeficient")
    cbar = plt.colorbar()
    #cbar.set_label("Intensity (dB)")
    plt.show()


