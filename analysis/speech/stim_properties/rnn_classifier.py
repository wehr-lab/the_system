# Import Libraries
import sys
import getopt
from time import strftime, gmtime

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout, PReLU, Input, merge
from keras.optimizers import RMSprop, Adam, Nadam
from keras.regularizers import l2
from keras.models import load_model, Sequential
import numpy as np

from learning_utils import *


PHOVECT_COLS = ['consonant', 'speaker', 'vowel', 'token']
NAMES = ['Jonny', 'Ira', 'Anna', 'Dani', 'Theresa']
CONS = ['g', 'b']
VOWS = ['I', 'o', 'a', 'ae', 'e', 'u']
PHONEME_DIR = '/home/lab/github/SVN-ComboPack/ratrixSounds/phonemes/'

MAPBACK = {'g':1,'b':2,
           'Jonny':1,'Ira':2,'Anna':3,'Dani':4,'Theresa':5,
           'I':1,'o':2,'a':3,'ae':4,'e':5,'u':6}

SPEC_TYPE = "spec" # or "spec"

MEL_PARAMS = {
    "n_fft"      : 960,
    "hop_length" : 960,
    "n_mels"     : 64,
    "fmin"       : 1000,
    "htk"        : True
}

SPEC_PARAMS = {
    "window"          : "hamming",
    "nperseg"         : 480,
    "noverlap"        : 0,
    "return_onesided" : True,
    "scaling"         : "spectrum"
}

LOAD_PARAMS = {
    "scale"      : True,
    "prop_train" : 0.9,
    "as_row"     : False,

    # What form should the data be in?
    # possible vals:
        #   "regression" - fit spectrograms to continuous predictions
        #   "trials"     - fit spectrograms to record of trials
    "net_type"   : "regression",

    # Weight training by what?
    # possible vals:
        #   None - don't weight
        #   "correct" - weight by accuracy
    "weight_by"  : None
}


# DATA_DIR       = '/Users/Jonny/Documents/fixSpeechData/'
LSTM_NETSIZE   = 128
FILT_LEN       = 3
LEARN_RATE     = 0.0001
L2_WEIGHT      = 0.00001
OPTIMIZER_TYPE = "adam" # nadam, adam, or rmsprop
NET_TYPE       = LOAD_PARAMS["net_type"]

# Jitter audio so all don't start at same time
N_JIT = 1
JIT_AMT = 0.1 # In seconds


##########################################

phovect_idx, phovect_xdi, file_idx, file_loc = (
    make_phoneme_iterators(NAMES, CONS, VOWS, MAPBACK, PHONEME_DIR)
    )

X_train,cons_train,speak_train,vow_train = spectrogram_for_training(file_loc, SPEC_PARAMS,phovect_idx)




##########################################
# Check if we were passed a model string or if we are training fresh
try:
    opts,args = getopt.getopt(sys.argv[1:],"p:")
except:
    opts = None
    args = None

loaded = False
for opt, arg in opts:
    if opt == '-p':
        print('Loading model {}'.format(arg))
        model = load_model(arg)
        loaded = True

if not loaded:
    print('Building model')

    if OPTIMIZER_TYPE == "nadam":
        optimizer = Nadam(lr=LEARN_RATE)
    elif OPTIMIZER_TYPE == "adam":
        optimizer = Adam(lr=LEARN_RATE)
    else:
        optimizer = RMSprop(lr=LEARN_RATE, clipnorm=1.)

    model = Sequential()
    model.add(LSTM(LSTM_NETSIZE,
    			   dropout_W=.4, dropout_U=0.2,
    			   init="he_normal",
    			   return_sequences=True,
    			   activation=PReLU(),
    			   batch_input_shape=))


	
