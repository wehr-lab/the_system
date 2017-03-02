
# Import Libraries
from python_speech_features import mfcc
import itertools
from scipy.io import wavfile
from scipy.signal import spectrogram, decimate
import pandas
import numpy as np
from librosa.feature import melspectrogram

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


###########################
# Declare Parameters
# Empirically derived parameters - in general speech sounds have a narrower FFT window that other sounds
# MFCC_PARAMS = {
#     "winlen"       : 0.005,
#     "winstep"      : 0.0025,
#     "numcep"       : 13,
#     "nfilt"        : 26,
#     "nfft"         : 512,
#     "lowfreq"      : 1000,
#     "preemph"      : 0,
#     "ceplifter"    : 0,
#     "appendEnergy" : False
# }
#
# MEL_PARAMS = {
#     "n_fft"      : 480,
#     "hop_length" : 240,
#     "n_mels"     : 20,
#     "fmin"       : 1000
# }
#
# PHOVECT_COLS = ['consonant','speaker','vowel','token']
# NAMES = ['Jonny','Ira','Anna','Dani','Theresa']
# CONS = ['g','b']
# VOWS = ['I','o','a','ae','e','u']
# #phoneme_dir = '/home/lab/github/SVN-ComboPack/ratrixSounds/phonemes/'
# PHONEME_DIR = '/Users/Jonny/Github/SVN-ComboPack/ratrixSounds/phonemes/'
#
# MAPBACK = {'g':1,'b':2,
#            'Jonny':1,'Ira':2,'Anna':3,'Dani':4,'Theresa':5,
#            'I':1,'o':2,'a':3,'ae':4,'e':5,'u':6}
#
# DATA_DIR = '/Users/Jonny/Documents/fixSpeechData/'
#
# PROP_TRAIN = 0.8 # Proportion of data to use as training data

###########################
# Load audio files & decompose into MFCC's
def mfcc_from_files(file_loc,MFCC_PARAMS):
    phonemes = dict()
    mfccs = dict()
    mfccs_np = np.ndarray((len(file_loc),2600))
    for i, loc in file_loc.items():
        fs, phonemes[i] = wavfile.read(loc)
        an_mfcc = mfcc(phonemes[i],fs,**MFCC_PARAMS)
        mfccs[i] = an_mfcc
        mfccs_np[i,0:an_mfcc.shape[0]*an_mfcc.shape[1]] = an_mfcc.reshape(-1)
    return mfccs,mfccs_np

def mel_from_files(file_loc, MEL_PARAMS):
    phonemes = dict()
    mels = dict()
    for i, loc in file_loc.items():
        fs, phonemes[i] = wavfile.read(loc)
        a_mel = melspectrogram(phonemes[i], sr=fs, **MEL_PARAMS)
        mels[i] = a_mel
    return mels


def spectrogram_from_files(file_loc, SPECT_PARAMS):
    phonemes = dict()
    specs = dict()
    for i, loc in file_loc.items():
        fs, phonemes[i] = wavfile.read(loc)
        f, t, a_spec = spectrogram(phonemes[i], fs=fs, **SPECT_PARAMS)
        specs[i] = a_spec
    return f, t, specs

def spectrogram_for_training(file_loc, SPECT_PARAMS, phovect_idx, n_jitter = 1, jitter_range = None):
    phonemes = dict()
    specs = dict()
    constype  = np.zeros((len(phovect_idx),2),dtype=np.bool)
    vowtype   = np.zeros((len(phovect_idx),6),dtype=np.bool)
    speaktype = np.zeros((len(phovect_idx),5),dtype=np.bool)
    maxlen = 0
    for i, loc in file_loc.items():
        fs, phoneme = wavfile.read(loc)
        a_spec = melspectrogram(phoneme, sr=fs, **SPECT_PARAMS)
        specs[i] = a_spec

        #constype[i] = np.bool(phovect_idx[i][0]-1)
        constype[i,(phovect_idx[i][0]-1)] = True

        speaktype[i,phovect_idx[i][1]-1] = True
        vowtype[i,phovect_idx[i][2]-1] = True



        # Make numpy array of sounds, and if requested, randomly jitter start time
    if n_jitter == 1:
        specs_np = np.zeros((len(specs),specs[0].shape[0],specs[0].shape[1],1),dtype=np.float)
        for i, spec in specs.items():
                specs_np[i,:,:,0] = spec
    else:
        # In this version, we jitter by an integer amount of time bins
        jitmax = np.int(jitter_range)
        specs_np  = np.zeros((len(specs) * n_jitter,specs[0].shape[0],
                                specs[0].shape[1]+jitmax,1),
                                dtype = np.int16)
        constype_np  = np.zeros(((len(specs) * n_jitter),2),
                                dtype = np.bool)
        vowtype_np   = np.zeros((len(specs) * n_jitter, 6),
                                dtype=np.bool)
        speaktype_np = np.zeros((len(specs) * n_jitter, 5),
                                dtype=np.bool)

        for i, spec in specs.items():
            for j in range(n_jitter):
                thisind = np.int(i*n_jitter + j)
                jitby = np.int(np.random.randint(0,jitmax))
                specs_np[thisind,0:spec.shape[0],jitby:(jitby+spec.shape[1]),0] = spec
                constype_np[thisind,(phovect_idx[i][0]-1)] = True
                #constype_np[thisind] = np.bool(phovect_idx[i][0]-1)
                speaktype_np[thisind,phovect_idx[i][1]-1] = True
                vowtype_np[thisind,phovect_idx[i][2]-1] = True

        # rename constype so we don't have multiple return conditions
        constype  = constype_np
        speaktype = speaktype_np
        vowtype   = vowtype_np



    randind = np.random.permutation(specs_np.shape[0])
    specs_np = specs_np[randind,:,:,:]
    constype = constype[randind,:]
    speaktype = speaktype[randind,:]
    vowtype = vowtype[randind,:]


    return specs_np,constype,speaktype,vowtype


def audio_from_files(file_loc,phovect_idx,n_jitter=1,jitter_range=.1):

    phonemes = dict()
    constype  = np.zeros((len(phovect_idx),2),dtype=np.float)
    vowtype   = np.zeros((len(phovect_idx),6),dtype=np.float)
    speaktype = np.zeros((len(phovect_idx),5),dtype=np.float)
    maxlen = 0
    for i, loc in file_loc.items():
        fs, sig = wavfile.read(loc)
        # phonemes[i] = decimate(sig,2,ftype="fir",zero_phase=True)
        phonemes[i] = sig

        # If /b/, col 0 True, if /g/, etc.
        constype[i,phovect_idx[i][0]-1] = 1.
        speaktype[i,phovect_idx[i][1]-1] = 1.
        vowtype[i,phovect_idx[i][2]-1] = 1.

        # Find the longest file if they're of different lengths
        if len(phonemes[i]) > maxlen:
            maxlen = len(phonemes[i])

    # Make numpy array of sounds, and if requested, randomly jitter start time
    if n_jitter == 1:
        phonemes_np = np.zeros((len(phonemes),maxlen),dtype=np.int16)
        for i, phone in phonemes.items():
            phonemes_np[i,range(len(phone))] = phone
    else:
        jitmax = np.int(np.round(jitter_range * fs))
        phonemes_np  = np.zeros((len(phonemes) * n_jitter,
                                maxlen + jitmax),
                                dtype = np.int16)
        constype_np  = np.zeros((len(phonemes) * n_jitter, 2),
                                dtype = np.float)
        vowtype_np   = np.zeros((len(phonemes) * n_jitter, 6),
                                dtype=np.float)
        speaktype_np = np.zeros((len(phonemes) * n_jitter, 5),
                                dtype=np.float)

        for i, phone in phonemes.items():
            for j in range(n_jitter):
                thisind = np.int(i*n_jitter + j)
                jitby = np.int(np.random.randint(0,jitmax))
                phonemes_np[thisind,range(jitby,jitby+len(phone))] = phone
                constype_np[thisind,phovect_idx[i][0]-1] = 1.
                speaktype_np[thisind,phovect_idx[i][1]-1] = 1.
                vowtype_np[thisind,phovect_idx[i][2]-1] = 1.

        # rename constype so we don't have multiple return conditions
        constype  = constype_np
        speaktype = speaktype_np
        vowtype   = vowtype_np


    # Randomize Order
    randind = np.random.permutation(phonemes_np.shape[0])
    phonemes_np = phonemes_np[randind,:]
    constype = constype[randind,:]
    speaktype = speaktype[randind,:]
    vowtype = vowtype[randind,:]

    # Add singleton dim to fit keras
    phonemes_np = phonemes_np.reshape(len(phonemes_np),phonemes_np.shape[1],1)

    return phonemes_np,constype,speaktype,vowtype




# Declare scalar token number from phovect [consonant,speaker, vowel,token]
# Using stimmap 1 {Jonny, Ira, Anna, Dani, Theresa}


def make_phoneme_iterators(names, cons, vows, mapback, phoneme_dir):
    # Make dicts to map phonemes to data and to audio files
    # phovect_idx = # -> tuple
    # phovect_xdi = tuple -> #
    # file_idx    = # -> file components
    # file_loc    = # -> file

    file_iterator = itertools.product(cons, names, vows, range(1, 4))
    counter = itertools.count()
    phovect_idx = dict()
    phovect_xdi = dict()
    file_idx = dict()
    file_loc = dict()
    while True:
        try:
            file_components = file_iterator.next()
            if ((file_components[1] == 'Jonny') & any(vow for vow in file_components[2] if vow in ['ae','e','u'])) or (file_components[0] == 'g' and file_components[1] == 'Anna' and file_components[2] is "ae" and file_components[3] == 3 ):
                pass
            else:
                thisnum = counter.next()
                pho = file_components[0] + file_components[2]
                phovect_tup = (mapback[file_components[0]],mapback[file_components[1]],mapback[file_components[2]],file_components[3])
                phovect_idx[thisnum] = phovect_tup
                phovect_xdi[phovect_tup] = thisnum
                file_idx[thisnum] = file_components
                file_loc[thisnum] = phoneme_dir + file_components[1] + '/CV/' + pho + '/' + pho + str(file_components[3]) + '.wav'
        except StopIteration:
            break
    return phovect_idx,phovect_xdi,file_idx,file_loc




# Load behavior data & get ndarray with pho vect
# correct answer & mouse's answer
# Behavior data should be as hdf5
# and should have already been cleaned for generalization (eg. fixDirtyGens)
# data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]

def load_mouse(f,col_select,specs,phovect_xdi,phovect_idx,path=None,scale=True,prop_train=0.8,as_row=True,net_type="regression",weight_by=None):
    # Specs is a collection of spectrogram-like ndarrays, like mfcc's or mel specs.
    # Can give full path as f or filename and path

    # None out optional returns
    weights = None
    X_test  = None
    y_test  = None

    if path:
        f = path+f

    h5f = pandas.HDFStore(f, mode="r")['table']

    ##################################################
    # Switch depending on what type of network we want
    if net_type == "trials":
        print('\n Making data for trial-fitting...')
        h5f.phovect = h5f[col_select]
        X_train = np.ndarray((h5f.phovect.__len__(),specs[0].shape[0],specs[0].shape[1]))

        # Make spectrogram array
        for i in range(0,h5f.phovect.__len__()):
            phovect = tuple(h5f.phovect.iloc[i])
            this_spec = specs[phovect_xdi[phovect]]
            X_train[i, 0:this_spec.shape[0],0:this_spec.shape[1]] = this_spec

        # Take out any nans and make responses array
        responses = h5f['response']
        response_nans = np.isnan(responses)
        keep_trials = response_nans[response_nans==False].index
        responses = responses[keep_trials]
        X_train = X_train[keep_trials,:,:]
        y_train = np.zeros((len(responses),2),dtype=np.bool)
        y_train[responses==0,0] = True
        y_train[responses==1,1] = True

    elif net_type == "regression":
        print('\n Making data for regression...')
        pho_response = h5f.groupby(by=col_select)['response'].mean()
        if weight_by == "correct":
            pho_meancx = h5f.groupby(by=col_select)['correct'].mean()
            weights    = np.zeros((pho_meancx.__len__()),dtype=np.float64)

        X_train = np.ndarray((pho_response.__len__(), specs[0].shape[0], specs[0].shape[1]))
        y_train = np.ndarray(pho_response.__len__(),dtype=np.float64)

        for i in phovect_idx.keys():
            X_train[i,0:specs[i].shape[0],0:specs[i].shape[1]] = specs[i]

            phovect = phovect_idx[i]
            y_train[i] = 100*pho_response[phovect]
            if weight_by == "correct":
                weights[i] = pho_meancx[phovect]

    ##################################################

    # Reshape & Rescale specs
    if scale is True:
        print('\n Scaling Spectrograms')
        X_train = X_train.reshape(X_train.shape[0], -1)
        preprocessing.scale(X_train.astype(float),axis=1,copy=False)

    # Split into training & test sets
    if net_type == "trials":
        print('\n Splitting Data')
        X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                            y_train,
                                                            train_size = prop_train)

    # If we want the spectrograms returned in 2D...
    if as_row == False:
        X_train = X_train.reshape(X_train.shape[0],
                                  specs[0].shape[0],
                                  specs[0].shape[1])
        if X_test:
            X_test = X_test.reshape(X_test.shape[0],
                                    specs[0].shape[0],
                                    specs[0].shape[1])

    # Randomize order
    randind = np.random.permutation(X_train.shape[0])
    X_train = X_train[randind,:,:]
    y_train = y_train[randind]
    if weights:
        weights = weights[randind]

    # Add singleton dimension to fit Keras & take off last column
    X_train = X_train.reshape(len(X_train),1,X_train.shape[1],X_train.shape[2])
    X_train = X_train[:,:,:,:-1]
    if X_test:
        X_test = X_test.reshape(len(X_test),1,X_test.shape[1],X_test.shape[2])
        X_test = X_test[:,:,:,:-1]

    return X_train,y_train,X_test,y_test,weights

