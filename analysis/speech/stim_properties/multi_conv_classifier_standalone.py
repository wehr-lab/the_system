
# Import Libraries
import sys
import getopt
from time import strftime, gmtime

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D, Dropout, ELU, Input, merge, BatchNormalization
from keras.optimizers import RMSprop, Adam, Nadam
from keras.regularizers import l2
from keras.models import load_model, Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, RemoteMonitor
import keras.backend as K
import numpy as np

from learning_utils import *
import threading

# Testing Hualos for visualization
#import api
#sthread = threading.Thread(target=server.serve_forever)
#sthread.setDaemon(True)
#sthread.start()


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
    "hop_length" : 480,
    "n_mels"     : 100,
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
N_CONV_FILTERS = 64
FILT_LEN       = 3
LEARN_RATE     = 0.0001
L2_WEIGHT      = 0.001
OPTIMIZER_TYPE = "adam" # nadam, adam, or rmsprop
NET_TYPE       = LOAD_PARAMS["net_type"]

# Jitter audio so all don't start at same time
N_JIT = 3
JIT_AMT = 10 # In seconds or time bins


##########################################

phovect_idx, phovect_xdi, file_idx, file_loc = (
    make_phoneme_iterators(NAMES, CONS, VOWS, MAPBACK, PHONEME_DIR)
    )

X_train,cons_train,speak_train,vow_train = spectrogram_for_training(file_loc, MEL_PARAMS,phovect_idx,N_JIT,JIT_AMT)

def concat_cons_speak(cons, speak):
    cat = np.zeros((speak.shape[0],10), dtype=np.bool)
    for i in range(speak.shape[0]):
        if cons[i,0]:
            cat[i,np.where(speak[i,:])[0][0]] = True
        else:
            cat[i, np.where(speak[i, :])[0][0]+5] = True

    return cat

conscat = concat_cons_speak(cons_train, speak_train)

##########################################
# Custom model functions/objects
def mean_pred(y_true,y_pred):
    return {
        'mean_in' : K.mean(y_true),
        'mean_out' : K.mean(y_pred)
    }

learn_drop = ReduceLROnPlateau(monitor="loss",
                               factor=0.1,
                               patience=10,
                               mode="min",
                               epsilon=0.1,
                               cooldown=5,
                               min_lr=(LEARN_RATE/1000),
                               verbose=1
                               )

checkpoint = ModelCheckpoint('/home/lab/Speech_Models/multi_mel_conv_conscat_E{epoch:02d}-L{loss:.2f}-conscat{acc:.2f}', monitor="val_loss")

remote = RemoteMonitor(root='http://localhost:9000')

#s_thread = threading.Thread(target=api.server.serve_forever)
#s_thread.start()


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
    l_conv_1_1 = Convolution2D(N_CONV_FILTERS, FILT_LEN, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            dim_ordering="tf",
                            W_regularizer=l2(L2_WEIGHT))(l_input)
    l_norm_1_1 = BatchNormalization(axis=3)(l_conv_1_1)
    l_act_1_1  = ELU()(l_norm_1_1)
    l_drop_1_1 = Dropout(0.5)(l_act_1_1)

    l_conv_1_2 = Convolution2D(N_CONV_FILTERS, FILT_LEN, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT))(l_drop_1_1)
    l_norm_1_2 = BatchNormalization(axis=3)(l_conv_1_2)
    l_act_1_2  = ELU()(l_norm_1_2)
    l_drop_1_2 = Dropout(0.5)(l_act_1_2)

    l_conv_1_3 = Convolution2D(N_CONV_FILTERS, FILT_LEN, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT))(l_drop_1_2)
    l_norm_1_3 = BatchNormalization(axis=3)(l_conv_1_3)
    l_act_1_3  = ELU()(l_norm_1_3)
    l_drop_1_3 = Dropout(0.5)(l_act_1_3)

    l_pool_1   = MaxPooling2D(pool_size=(2,2))(l_drop_1_3)

    # CONV LEVEL 2
    l_conv_2_1 = Convolution2D(N_CONV_FILTERS*2, FILT_LEN, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT))(l_pool_1)
    l_norm_2_1 = BatchNormalization(axis=3)(l_conv_2_1)
    l_act_2_1  = ELU()(l_norm_2_1)
    l_drop_2_1 = Dropout(0.5)(l_act_2_1)

    l_conv_2_2 = Convolution2D(N_CONV_FILTERS*2, FILT_LEN, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT))(l_drop_2_1)
    l_norm_2_2 = BatchNormalization(axis=3)(l_conv_2_2)
    l_act_2_2  = ELU()(l_norm_2_2)
    l_drop_2_2 = Dropout(0.5)(l_act_2_2)

    l_conv_2_3 = Convolution2D(N_CONV_FILTERS*2, FILT_LEN, FILT_LEN,
                            init='he_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT))(l_drop_2_2)
    l_norm_2_3 = BatchNormalization(axis=3)(l_conv_2_3)
    l_act_2_3  = ELU()(l_norm_2_3)
    l_drop_2_3 = Dropout(0.5)(l_act_2_3)

    l_pool_2   = MaxPooling2D(pool_size=(2,2))(l_drop_2_3)

    # CONV LEVEL 3
    # l_conv_3_1 = Convolution2D(N_CONV_FILTERS*4, FILT_LEN, FILT_LEN,
    #                         init='he_normal',
    #                         border_mode='same',
    #                         W_regularizer=l2(L2_WEIGHT))(l_pool_2)
    # l_norm_3_1 = BatchNormalization(axis=3)(l_conv_3_1)
    # l_act_3_1  = ELU()(l_norm_3_1)
    # l_drop_3_1 = Dropout(0.3)(l_act_3_1)

    # l_conv_3_2 = Convolution2D(N_CONV_FILTERS*4, FILT_LEN, FILT_LEN,
    #                         init='he_normal',
    #                         border_mode='same',
    #                         W_regularizer=l2(L2_WEIGHT))(l_drop_3_1)
    # l_norm_3_2 = BatchNormalization(axis=3)(l_conv_3_2)
    # l_act_3_2  = ELU()(l_norm_3_2)
    # l_drop_3_2 = Dropout(0.3)(l_act_3_2)
    # l_pool_3   = MaxPooling2D(pool_size=(2,2))(l_drop_3_2)


    # # DENSE LAYERS
    l_flat     = Flatten()(l_pool_2)

    l_dense_1  = Dense(N_CONV_FILTERS,
                       init="he_normal",
                       W_regularizer=l2(L2_WEIGHT))(l_flat)
    l_dnorm_1 = BatchNormalization(axis=-1)(l_dense_1)
    l_dact_1   = ELU()(l_dnorm_1)
    l_ddrop_1  = Dropout(0.5)(l_dact_1)

    # l_dense_2  = Dense(N_CONV_FILTERS,
    #                    init="he_normal",
    #                    W_regularizer=l2(L2_WEIGHT))(l_flat)
    # l_dnorm_2 = BatchNormalization(axis=-1)(l_dense_2)
    # l_dact_2   = ELU()(l_dnorm_2)
    # l_ddrop_2  = Dropout(0.5)(l_dact_2)
    #
    # l_dense_3  = Dense(N_CONV_FILTERS,
    #                    init="he_normal",
    #                    W_regularizer=l2(L2_WEIGHT))(l_flat)
    # l_dnorm_3 = BatchNormalization(axis=-1)(l_dense_3)
    # l_dact_3   = ELU()(l_dnorm_3)
    # l_ddrop_3  = Dropout(0.5)(l_dact_3)
    # l_dense_2  = Dense(N_CONV_FILTERS*2,activation="tanh",
    #                    init="he_normal",
    #                    W_regularizer=l2(L2_WEIGHT))(l_ddrop_1)
    # l_ddrop_2  = Dropout(0.3)(l_dense_2)

    # OUTPUTS
    # l_out_speak = Dense(5,activation='softmax',
    #                     init="he_normal",
    #                     name="speak",
    #                     W_regularizer=l2(L2_WEIGHT))(l_flat)
    # l_out_vow   = Dense(6,activation='softmax',
    #                     init="he_normal",
    #                     name="vow",
    #                     W_regularizer=l2(L2_WEIGHT))(l_flat)

    # l_merge     = merge([l_flat,l_out_speak,l_out_vow],mode='concat',concat_axis=1)
    # l_dense_2   = Dense(N_CONV_FILTERS*2,activation="relu",
    #                     init="he_normal",
    #                     W_regularizer=l2(L2_WEIGHT))(l_merge)
    # l_drop_2    = Dropout(0.5)(l_dense_2)

    l_out_cons  = Dense(10,activation='sigmoid',
                        init="he_normal",
                        name="cons_spk",
                        W_regularizer=l2(L2_WEIGHT))(l_ddrop_1)
    # l_out_speak = Dense(5,activation='sigmoid',
    #                     init="he_normal",
    #                     name="speak",
    #                     W_regularizer=l2(L2_WEIGHT))(l_ddrop_2)
    # l_out_vow   = Dense(6,activation='sigmoid',
    #                     init="he_normal",
    #                     name="vow",
    #                     W_regularizer=l2(L2_WEIGHT))(l_ddrop_3)


    # COMPILE MODEL
    model = Model(input=[l_input],
                  output=[l_out_cons])
                  #output=[l_out_cons])
    model.compile(optimizer=optimizer,
                  #loss=['categorical_crossentropy',
                  #      'categorical_crossentropy',
                  #      'categorical_crossentropy'],
                  #loss_weights=[1., 0.25, 0.25],
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

for i in range(5):
    X_train,cons_train,speak_train,vow_train = spectrogram_for_training(file_loc, MEL_PARAMS,phovect_idx,N_JIT,JIT_AMT)
    conscat = concat_cons_speak(cons_train, speak_train)
        #model.fit(X_train, [cons_train, speak_train, vow_train], batch_size=5, nb_epoch=1)
    model.fit(X_train, conscat, batch_size=10, nb_epoch=300,callbacks=[learn_drop,checkpoint,remote])


