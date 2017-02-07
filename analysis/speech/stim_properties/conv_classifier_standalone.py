
# Import Libraries
import sys,getopt

from python_speech_features import mfcc
import itertools
from scipy.io import wavfile
from scipy.stats import mode
from scipy.signal import spectrogram
import pandas
import numpy as np
from librosa.feature import melspectrogram
from time import strftime,gmtime

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, advanced_activations, Convolution2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import RMSprop, Adam, Nadam
from keras.regularizers import l2,activity_l2
from keras.models import load_model

from learning_utils import *


PHOVECT_COLS = ['consonant', 'speaker', 'vowel', 'token']
NAMES = ['Jonny', 'Ira', 'Anna', 'Dani', 'Theresa']
CONS = ['g', 'b']
VOWS = ['I', 'o', 'a', 'ae', 'e', 'u']
PHONEME_DIR = '/Users/Jonny/Github/SVN-ComboPack/ratrixSounds/phonemes/'

MAPBACK = {'g':1,'b':2,
           'Jonny':1,'Ira':2,'Anna':3,'Dani':4,'Theresa':5,
           'I':1,'o':2,'a':3,'ae':4,'e':5,'u':6}

SPEC_TYPE = "mel" # or "spec"

MEL_PARAMS = {
    "n_fft"      : 960,
    "hop_length" : 960,
    "n_mels"     : 64,
    "fmin"       : 1000,
    "htk"        : True
}

SPEC_PARAMS = {
    "window"          : "hamming",
    "nperseg"         : 960,
    "noverlap"        : 480,
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


DATA_DIR       = '/Users/Jonny/Documents/fixSpeechData/'
N_CONV_FILTERS = 128
LEARN_RATE     = 0.0001
L2_WEIGHT      = 0.0001
OPTIMIZER_TYPE = "nadam" # nadam, adam, or rmsprop
NET_TYPE       = LOAD_PARAMS["net_type"]


##########################################
# Load data for 1 mouse
f = '/Users/Jonny/Documents/fixSpeechData/6928.h5'

phovect_idx, phovect_xdi, file_idx, file_loc = (
    make_phoneme_iterators(NAMES, CONS, VOWS, MAPBACK, PHONEME_DIR)
    )

if SPEC_TYPE == "mel":
    spects = mel_from_files(file_loc, MEL_PARAMS)
elif SPEC_TYPE == "spec":
    spects = spectrogram_from_files(file_loc,SPEC_PARAMS)


X_train, y_train, X_test, y_test, weights = (
    load_mouse(f,PHOVECT_COLS,spects,phovect_xdi,phovect_idx,
               **LOAD_PARAMS)
    )




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
        optimizer = Nadam(lr=LEARN_RATE, clipnorm=1.)
    elif OPTIMIZER_TYPE == "adam":
        optimizer = Adam(lr=LEARN_RATE, clipnorm=1.)
    else:
        optimizer = RMSprop(lr=LEARN_RATE, clipnorm=1.)

    #Build model
    #Convolution layers
    #As per https://arxiv.org/pdf/1412.6806v3.pdf
    model = Sequential()
    # FIRST 2 CONV LAYERS
    model.add(Convolution2D(N_CONV_FILTERS, 3, 3,
                            init='glorot_normal',
                            border_mode='same',
                            dim_ordering='th',
                            W_regularizer=l2(L2_WEIGHT),
                            activation='relu',
                            #activity_regularizer=activity_l2(L2_WEIGHT),
                            input_shape=(1,X_train.shape[2],X_train.shape[3])))
    model.add(Dropout(0.1))
    model.add(Convolution2D(N_CONV_FILTERS, 3, 3,
                            init='glorot_normal',
                            border_mode='same',
                            activation='relu',
                            W_regularizer=l2(L2_WEIGHT),
                            #activity_regularizer=activity_l2(L2_WEIGHT),
                            dim_ordering='th'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(N_CONV_FILTERS, 3, 3,
                            init='glorot_normal',
                            border_mode='same',
                            activation='relu',
                            W_regularizer=l2(L2_WEIGHT),
                            #activity_regularizer=activity_l2(L2_WEIGHT),
                            dim_ordering='th'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(N_CONV_FILTERS * 2, 3, 3,
                            init='glorot_normal',
                            border_mode='same',
                            dim_ordering='th',
                            activation='relu',
                            #activity_regularizer=activity_l2(L2_WEIGHT),
                            W_regularizer=l2(L2_WEIGHT)))
    model.add(Dropout(0.1))
    model.add(Convolution2D(N_CONV_FILTERS * 2, 3, 3,
                            init='glorot_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT),
                            #activity_regularizer=activity_l2(L2_WEIGHT),
                            activation='relu',
                            dim_ordering='th'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(N_CONV_FILTERS * 4, 3, 3,
                            init='glorot_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT),
                            activation='relu',
                            dim_ordering='th'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(N_CONV_FILTERS * 4, 3, 3,
                            init='glorot_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT),
                            activation='relu',
                            dim_ordering='th'))
    model.add(Dropout(0.1))
    model.add(Flatten())
#   model.add(Dense(N_CONV_FILTERS*2,
#                   activation='tanh',
#                   activity_regularizer=activity_l2(L2_WEIGHT*10),
#                   ))
#   model.add(Dropout(0.3))

if NET_TYPE == "regression":
    model.add(Dense(1, activation='linear'))
    model.add(Dropout(0.2))
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error'])
    for i in range(100):
        model.fit(X_train, y_train, batch_size=1, nb_epoch=5)
        model_str = '/Users/Jonny/Documents/Speech_Models/conv_reg_{}_N{}'.format(strftime("%m%d%H%M", gmtime()),N_CONV_FILTERS)
        model.save(model_str)

elif NET_TYPE == "trials":
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    for i in range(100):
        model.fit(X_train, y_train, batch_size=128, nb_epoch=5)
        model_str = '/Users/Jonny/Documents/Speech_Models/conv_trials_{}_N{}'.format(strftime("%m%d%H%M", gmtime()),N_CONV_FILTERS)
        model.save(model_str)
