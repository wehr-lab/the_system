
# Import Libraries
import sys,getopt

from python_speech_features import mfcc
import itertools
from scipy.io import wavfile
import pandas
import numpy as np
from librosa.feature import melspectrogram

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, advanced_activations, Convolution2D, Flatten, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras.models import load_model


PHOVECT_COLS = ['consonant', 'speaker', 'vowel', 'token']
NAMES = ['Jonny', 'Ira', 'Anna', 'Dani', 'Theresa']
CONS = ['g', 'b']
VOWS = ['I', 'o', 'a', 'ae', 'e', 'u']
PHONEME_DIR = '/Users/Jonny/Github/SVN-ComboPack/ratrixSounds/phonemes/'

MAPBACK = {'g':1,'b':2,
           'Jonny':1,'Ira':2,'Anna':3,'Dani':4,'Theresa':5,
           'I':1,'o':2,'a':3,'ae':4,'e':5,'u':6}

MEL_PARAMS = {
    "n_fft"      : 960,
    "hop_length" : 480,
    "n_mels"     : 30,
    "fmin"       : 1500
}

# Plot mel filterbank
#melfb = filters.mel(96000,1500,40,1500)
#plt.figure()
#display.specshow(melfb,x_axis='linear')

DATA_DIR       = '/Users/Jonny/Documents/fixSpeechData/'
PROP_TRAIN     = 0.9 # Proportion of data to use as training data
N_CONV_FILTERS = 100
LEARN_RATE     = 0.001
L2_WEIGHT      = 0.001

####################################

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




##########################################
# Load data for 1 mouse
f = '/Users/Jonny/Documents/fixSpeechData/6928.h5'

phovect_idx, phovect_xdi, file_idx, file_loc = make_phoneme_iterators(NAMES, CONS, VOWS, MAPBACK, PHONEME_DIR)
mels,        mels_np                         = mel_from_files(file_loc, MEL_PARAMS)
X_train,     X_test,      y_train,  y_test   = load_mouse(f,PHOVECT_COLS,mels,phovect_xdi,as_row=False)
X_train = X_train.reshape(len(X_train),1,X_train.shape[1],X_train.shape[2])
X_test = X_test.reshape(len(X_test),1,X_test.shape[1],X_test.shape[2])

# Take off last column
X_train = X_train[:,:,:,:-1]
X_test  = X_test[:,:,:,:-1]


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
        print('Loading model {}'.arg)
        model = load_model(arg)
        loaded = True

if not loaded:
    print('Building model')
    #Build model
    #Convolution layers
    #As per https://arxiv.org/pdf/1412.6806v3.pdf
    model = Sequential()
    # FIRST 2 CONV LAYERS
    model.add(Convolution2D(N_CONV_FILTERS,3,3,
                            init='glorot_normal',
                            border_mode='same',
                            dim_ordering='th',
                            W_regularizer=l2(L2_WEIGHT),
                            input_shape=(1,X_train.shape[2],X_train.shape[3])))
    model.add(Convolution2D(N_CONV_FILTERS,3,3,
                            init='glorot_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT),
                            dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(N_CONV_FILTERS,3,3,
                            init='glorot_normal',
                            border_mode='same',
                            dim_ordering='th',
                            W_regularizer=l2(L2_WEIGHT)))
    model.add(Convolution2D(N_CONV_FILTERS,3,3,
                            init='glorot_normal',
                            border_mode='same',
                            W_regularizer=l2(L2_WEIGHT),
                            dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(2,activation='softmax'))

    optimizer = RMSprop(lr=LEARN_RATE)
    #optimizer = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])







for i in range(30):
    model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=64, nb_epoch=5)
    model_str = '/Users/Jonny/Documents/Speech_Models/conv_{}_N{}'.format(strftime("%m%d%H%M", gmtime()),N_CONV_FILTERS)
    model.save(model_str)


model.save(model_str)

model.predict()



