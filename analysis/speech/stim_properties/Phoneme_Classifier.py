__author__ = 'Jonny Saunders'
###############################
# Train a random forest to emulate a mouse's decision function on the MFCC's of each token
###############################

# Import Libraries
from python_speech_features import mfcc
import itertools
from scipy.io import wavfile
import pandas
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

###########################
# Declare Parameters
# Empirically derived parameters - in general speech sounds have a narrower FFT window that other sounds
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

names = ['Jonny','Ira','Anna','Dani','Theresa']
cons = ['g','b']
vows = ['I','o','a','ae','e','u']
#phoneme_dir = '/home/lab/github/SVN-ComboPack/ratrixSounds/phonemes/'
phoneme_dir = '/Users/Jonny/Github/SVN-ComboPack/ratrixSounds/phonemes/'

mapback = {'g':1,'b':2,
           'Jonny':1,'Ira':2,'Anna':3,'Dani':4,'Theresa':5,
           'I':1,'o':2,'a':3,'ae':4,'e':5,'u':6}

data_dir = '/Users/Jonny/Documents/fixSpeechData/'



###########################

# Declare scalar token number from phovect [consonant,speaker, vowel,token]
# Using stimmap 1 {Jonny, Ira, Anna, Dani, Theresa}


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

# Load audio files & decompose into MFCC's
phonemes = dict()
mfccs = dict()
for i, loc in file_loc.items():
    fs, phonemes[i] = wavfile.read(loc)
    mfccs[i] = mfcc(phonemes[i],fs,**MFCC_PARAMS)


# Load behavior data & get ndarray with pho vect, correct answer & mouse's answer
# Behavior data should be as hdf5, and should have already been cleaned for generalization (eg. fixDirtyGens)
data_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

data_arrays = dict()
vect_arrays = dict()
response_arrays = dict()
mfcc_arrays = dict()

#for f in data_files: # For now just one mouse at a time thx
f = data_files[1]
h5f = pandas.HDFStore(data_dir + f, mode="r")['table']
phovect_cols = ['consonant','speaker','vowel','token']
h5f.phovect = h5f[phovect_cols]
mfcc_trials = np.ndarray((h5f.phovect.__len__(),200,13))
for i in range(0,h5f.phovect.__len__()):
    phovect = tuple(h5f.phovect.iloc[i])
    this_mfcc = mfccs[phovect_xdi[phovect]]
    test = np.split(this_mfcc, this_mfcc.shape[1], axis=1) # Make 2d array a list of vectors
    mfcc_trials[i, 0:this_mfcc.shape[0],0:this_mfcc.shape[1]] = this_mfcc

test = np.array(this_mfcc[0])
for i in this_mfcc:


# Get responses
responses = h5f['response']

# Take out any nans & convert responses to 0 and 1
response_nans = np.isnan(responses)
keep_trials = response_nans[response_nans==False].index
responses = responses[keep_trials]
mfcc_trials = mfcc_trials[keep_trials,:,:]
responses[responses==1] = 0
responses[responses==3] = 1

# Custom class so we can train on all the mfcc features at once
class FeatureSelector(BaseEstimator,TransformerMixin):
    def __init__(self,cep):
        self.cep = cep

    def fit(self,x,y=None):
        return self

    def transform(self,learn_stx):
        return learn_stx[:,:,self.cep]

pipeline = Pipeline([
    ('union',FeatureUnion(
        transformer_list=[
            ('cep0', FeatureSelector(cep=0)),
            ('cep1', FeatureSelector(cep=1))
        ]
    )),
    ('rfc',RandomForestClassifier(n_estimators=25))
])

pipeline.fit(responses,test2)


# Get sklernin on some trees
clf = RandomForestClassifier(n_estimators=25,n_jobs=4)
clf.fit(responses,mfcc_trials)

# TODO: IF length of arrays different, alert me
# TODO: Fix '7012' and '7007' because they have bullshit phovects.





# TODO: remember to convert responses to 0 and 1. Make feature_arrays by matching tuple in data_arrays to phovect_ind
    # and indexing the mfccs, make response arrays by just taking the list, then use as input to classifier

