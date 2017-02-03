
# Import Libraries
from python_speech_features import mfcc
import itertools
from scipy.io import wavfile
import pandas
import numpy as np
from librosa.feature import melspectrogram

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from treeinterpreter import treeinterpreter as ti

import the_system

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

# Declare scalar token number from phovect [consonant,speaker, vowel,token]
# Using stimmap 1 {Jonny, Ira, Anna, Dani, Theresa}

def make_phoneme_iterators(names,cons,vows,mapback,phoneme_dir):
    # Make dicts to map phonemes to data and to audio files
    # phovect_idx = # -> tuple
    # phovect_xdi = tuple -> #
    # file_idx    = # -> file components
    # file_loc    = # -> file


    file_iterator = itertools.product(cons,names,vows,range(1,4))
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

# Or just mel spectrum
def mel_from_files(file_loc,MEL_PARAMS):
    phonemes = dict()
    mels = dict()
    mels_np = np.ndarray((len(file_loc),4020))
    for i, loc in file_loc.items():
        fs, phonemes[i] = wavfile.read(loc)
        a_mel = melspectrogram(phonemes[i],sr=fs,**MEL_PARAMS)
        mels[i] = a_mel
        mels_np[i,0:a_mel.shape[0]*a_mel.shape[1]] = a_mel.reshape(-1)
    return mels,mels_np


# Load behavior data & get ndarray with pho vect, correct answer & mouse's answer
# Behavior data should be as hdf5, and should have already been cleaned for generalization (eg. fixDirtyGens)
#data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]

def load_mouse(f,col_select,specs,phovect_xdi,path=None,scale=True,prop_train=0.8,as_row=True):
    # Specs is a collection of spectrogram-like ndarrays, like mfcc's or mel specs.
    # Can give full path as f or filename and path
    if path:
        f = path+f

    data_arrays = dict()
    vect_arrays = dict()
    response_arrays = dict()
    mfcc_arrays = dict()

    h5f = pandas.HDFStore(f, mode="r")['table']
    h5f.phovect = h5f[col_select]

    print('\n Making Spectrogram Array...')
    # Make an array of spectrograms: trials x # frequencies in spec x # time points in spec
    spec_trials = np.ndarray((h5f.phovect.__len__(),specs[0].shape[0],specs[0].shape[1]))
    for i in range(0,h5f.phovect.__len__()):
        phovect = tuple(h5f.phovect.iloc[i])
        this_spec = specs[phovect_xdi[phovect]]
        spec_trials[i, 0:this_spec.shape[0],0:this_spec.shape[1]] = this_spec

    # Get responses
    print('\n Loading Responses')
    responses = h5f['response']

    # Take out any nans & convert responses to 0 and 1
    response_nans = np.isnan(responses)
    keep_trials = response_nans[response_nans==False].index
    responses = responses[keep_trials]
    spec_trials = spec_trials[keep_trials,:,:]
    #responses[responses==1] = 0
    #responses[responses==3] = 1
    resp_np = np.zeros((len(responses),2),dtype=np.bool)
    resp_np[responses==1,0] = True
    resp_np[responses!=1,1] = True
    responses = resp_np


    # Reshape & Rescale specs for concatenation
    spec_trials = spec_trials.reshape(len(responses), -1)

    if scale is True:
        print('\n Scaling Spectrograms')
        preprocessing.scale(spec_trials.astype(float),axis=1,copy=False)

    # Split into training & test sets
    print('\n Splitting Data')
    spec_train,spec_test,responses_train,responses_test = train_test_split(spec_trials,responses,train_size=prop_train)
    #spec_train = spec_trials
    #responses_train = responses
    #responses_test = None
    #spec_test = None

    if as_row == False:
        spec_train = spec_train.reshape(len(spec_train),this_spec.shape[0],this_spec.shape[1])
        spec_test  = spec_test.reshape(len(spec_test), this_spec.shape[0], this_spec.shape[1])

    return spec_train,spec_test,responses_train,responses_test
