
# Import Libraries
import sys
import getopt
from time import strftime, gmtime

from keras.models import Sequential
from keras.layers import Dense, Convolution1D, Flatten, MaxPooling1D, Dropout, PReLU, Input
from keras.optimizers import RMSprop, Adam, Nadam
from keras.regularizers import l2
from keras.models import load_model, Model
import numpy as np

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
N_CONV_FILTERS = 64
FILT_LEN       = 32
LEARN_RATE     = 0.001
L2_WEIGHT      = 0.0001
OPTIMIZER_TYPE = "adam" # nadam, adam, or rmsprop
NET_TYPE       = LOAD_PARAMS["net_type"]

# Jitter audio so all don't start at same time
N_JIT = 1
JIT_AMT = 0.1 # In seconds


##########################################

phovect_idx, phovect_xdi, file_idx, file_loc = (
    make_phoneme_iterators(NAMES, CONS, VOWS, MAPBACK, PHONEME_DIR)
    )

X_train,cons_train,speak_train,vow_train = audio_from_files(file_loc, phovect_idx, N_JIT, JIT_AMT)




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

    #Build model
    #Convolution layers
    #As per https://arxiv.org/pdf/1412.6806v3.pdf

    # CONV LEVEL 1
    l_input = Input(batch_shape=X_train.shape)
    l_conv_1_1 = Convolution1D(N_CONV_FILTERS, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT),
                            activation='relu')(l_input)
    l_drop_1_1 = Dropout(0.1)(l_conv_1_1)
    l_conv_1_2 = Convolution1D(N_CONV_FILTERS, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            activation='relu',
                            W_regularizer=l2(L2_WEIGHT))(l_drop_1_1)
    l_drop_1_2 = Dropout(0.1)(l_conv_1_2)
    l_pool_1   = MaxPooling1D(pool_length=8)(l_drop_1_2)

    # CONV LEVEL 2
    l_conv_2_1 = Convolution1D(N_CONV_FILTERS*2, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT),
                            activation='relu')(l_pool_1)
    l_drop_2_1 = Dropout(0.1)(l_conv_2_1)
    l_conv_2_2 = Convolution1D(N_CONV_FILTERS*2, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            activation='relu',
                            W_regularizer=l2(L2_WEIGHT))(l_drop_2_1)
    l_drop_2_2 = Dropout(0.1)(l_conv_2_2)
    l_pool_2   = MaxPooling1D(pool_length=8)(l_drop_2_2)

    # CONV LEVEL 3
    l_conv_3_1 = Convolution1D(N_CONV_FILTERS*2, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT),
                            activation='relu')(l_pool_2)
    l_drop_3_1 = Dropout(0.1)(l_conv_3_1)
    l_conv_3_2 = Convolution1D(N_CONV_FILTERS*2, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            activation='relu',
                            W_regularizer=l2(L2_WEIGHT))(l_drop_3_1)
    l_drop_3_2 = Dropout(0.1)(l_conv_3_2)

    # OUTPUTS
    l_flat     = Flatten()(l_drop_3_2)

    l_out_cons  = Dense(2,activation='softmax',
                        init="he_normal",
                        name="cons",
                        W_regularizer=l2(L2_WEIGHT))(l_flat)
    l_out_speak = Dense(5,activation='softmax',
                        init="he_normal",
                        name="speak",
                        W_regularizer=l2(L2_WEIGHT))(l_flat)
    l_out_vow   = Dense(6,activation='softmax',
                        init="he_normal",
                        name="vow",
                        W_regularizer=l2(L2_WEIGHT))(l_flat)

    # COMPILE MODEL
    model = Model(input=[l_input],
                  output=[l_out_cons,l_out_speak,l_out_vow])
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy'],
                  metrics=['accuracy'],
                  loss_weights=[1., 0.5, 0.5])

    for i in range(100):
        X_train,cons_train,speak_train,vow_train = audio_from_files(file_loc, phovect_idx, N_JIT, JIT_AMT)
        model.fit(X_train, [cons_train, speak_train, vow_train], batch_size=2, nb_epoch=1)
        model_str = '/Users/Jonny/Documents/Speech_Models/raw_multi_conv_{}_N{}'.format(strftime("%m%d%H%M", gmtime()),N_CONV_FILTERS)
        model.save(model_str)


