from time import strftime,gmtime

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, advanced_activations, Convolution2D, Flatten, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras.models import load_model


###########################
# Declare Parameters

MFCC_PARAMS = {
    "winlen"       : 0.005,
    "winstep"      : 0.0025,
    "numcep"       : 13,
    "nfilt"        : 26,
    "nfft"         : 512,
    "lowfreq"      : 1000,
    "preemph"      : 0,
    "ceplifter"    : 0,
    "appendEnergy" : False
}

MEL_PARAMS = {
    "n_fft"      : 960,
    "hop_length" : 480,
    "n_mels"     : 20,
    "fmin"       : 1000
}

# Plot mel filterbank
#melfb = filters.mel(96000,1500,40,1500)
#plt.figure()
#display.specshow(melfb,x_axis='linear')

PHOVECT_COLS = ['consonant', 'speaker', 'vowel', 'token']
NAMES = ['Jonny', 'Ira', 'Anna', 'Dani', 'Theresa']
CONS = ['g', 'b']
VOWS = ['I', 'o', 'a', 'ae', 'e', 'u']
PHONEME_DIR = '~/github/SVN-ComboPack/ratrixSounds/phonemes/'

MAPBACK = {'g':1,'b':2,
           'Jonny':1,'Ira':2,'Anna':3,'Dani':4,'Theresa':5,
           'I':1,'o':2,'a':3,'ae':4,'e':5,'u':6}

# DATA_DIR = '/Users/Jonny/Documents/fixSpeechData/'

# PROP_TRAIN = 0.9 # Proportion of data to use as training data

N_CONV_FILTERS = 256


##########################################
# Load data for 1 mouse
# f = '/Users/Jonny/Documents/fixSpeechData/6928.h5'

phovect_idx, phovect_xdi, file_idx, file_loc = make_phoneme_iterators(NAMES, CONS, VOWS, MAPBACK, PHONEME_DIR)
mels,        mels_np                         = mel_from_files(file_loc, MEL_PARAMS)
X_train,     X_test,      y_train,  y_test   = load_mouse(f,PHOVECT_COLS,mels,phovect_xdi,as_row=False)
X_train = X_train.reshape(len(X_train),1,X_train.shape[1],X_train.shape[2])
X_test = X_test.reshape(len(X_test),1,X_test.shape[1],X_test.shape[2])

# Take off last column
X_train = X_train[:,:,:,:-1]
X_test  = X_test[:,:,:,:-1]


##########################################
# Build model
# Convolution layers
# As per https://arxiv.org/pdf/1412.6806v3.pdf
model = Sequential()
# FIRST 2 CONV LAYERS
model.add(Convolution2D(N_CONV_FILTERS,3,3,
                        init='glorot_normal',
                        border_mode='same',
                        dim_ordering='th',
                        W_regularizer=l2(0.001),
                        input_shape=(1,X_train.shape[2],X_train.shape[3])))
#model.add(advanced_activations.ELU())
model.add(Convolution2D(N_CONV_FILTERS,3,3,
                        init='glorot_normal',
                        border_mode='same',
                        W_regularizer=l2(0.001),
                        dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(N_CONV_FILTERS,3,3,
                        init='glorot_normal',
                        border_mode='same',
                        dim_ordering='th',
                        W_regularizer=l2(0.001)))
#model.add(advanced_activations.ELU())
model.add(Convolution2D(N_CONV_FILTERS,3,3,
                        init='glorot_normal',
                        border_mode='same',
                        W_regularizer=l2(0.001),
                        dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Convolution2D(N_CONV_FILTERS,3,3,
#                        init='glorot_normal',
#                        border_mode='same',
#                        dim_ordering='th'))
#model.add(advanced_activations.ELU())
# DOWNSAMPLE
#model.add(Convolution2D(N_CONV_FILTERS,3,3,
#                        init='glorot_normal',
#                        border_mode='same',
#                        dim_ordering='th',
#                        W_regularizer=l2(0.001),
#                        subsample=(2,2))) # Stride of 2,2 to downsample)

# SECOND 2 CONV LAYERS w/ 2x N FILTERS
#model.add(Convolution2D(N_CONV_FILTERS,3,3,
#                        init='glorot_normal',
#                        border_mode='same',
#                        W_regularizer=l2(0.001),
#                        dim_ordering='th'))
#model.add(Convolution2D(N_CONV_FILTERS,3,3,
#                        init='glorot_normal',
#                        border_mode='same',
#                        dim_ordering='th'))
# DOWNSAMPLE 2
#model.add(Convolution2D(N_CONV_FILTERS,3,3,
#                        init='glorot_normal',
#                        border_mode='same',
#                       dim_ordering='th',
#                       W_regularizer=l2(0.001),
#                       subsample=(2,2))) # Stride of 2,2 to downsample)
# FINAL CONV
#model.add(Convolution2D(N_CONV_FILTERS*2,3,3,
#                        init='glorot_normal',
#                        border_mode='same',
#                        dim_ordering='th'))
# 1x1 CONVOLUTIONS
#model.add(Convolution2D(N_CONV_FILTERS,1,1,
#                        init='glorot_normal',
#                        border_mode='same',
#                        dim_ordering='th'))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))

optimizer = RMSprop(lr=0.005)
#optimizer = Adam(lr=0.001)


model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=64, nb_epoch=5)

model_str = '/Users/Jonny/Documents/Speech_Models/conv_{}_N{}'.format(strftime("%m%d%H%M", gmtime()),N_CONV_FILTERS)
model.save(model_str)

model.predict()





