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
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from treeinterpreter import treeinterpreter as ti
from sklearn import preprocessing
from librosa.feature import melspectrogram
from hmmlearn import hmm

from sklearn.datasets import load_boston

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
mfccs_np = np.ndarray((len(file_loc),2600))
for i, loc in file_loc.items():
    fs, phonemes[i] = wavfile.read(loc)
    an_mfcc = mfcc(phonemes[i],fs,**MFCC_PARAMS)
    mfccs[i] = an_mfcc
    mfccs_np[i,0:an_mfcc.shape[0]*an_mfcc.shape[1]] = an_mfcc.reshape(-1)

# Or just mel spectrum
phonemes = dict()
mfccs = dict()
mfccs_np = np.ndarray((len(file_loc),4020))
for i, loc in file_loc.items():
    fs, phonemes[i] = wavfile.read(loc)
    an_mfcc = melspectrogram(phonemes[i],sr=fs,n_fft=480,hop_length=240,n_mels=20,fmin=1000)
    mfccs[i] = an_mfcc
    mfccs_np[i,0:an_mfcc.shape[0]*an_mfcc.shape[1]] = an_mfcc.reshape(-1)


# Load behavior data & get ndarray with pho vect, correct answer & mouse's answer
# Behavior data should be as hdf5, and should have already been cleaned for generalization (eg. fixDirtyGens)
data_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

data_arrays = dict()
vect_arrays = dict()
response_arrays = dict()
mfcc_arrays = dict()

#for f in data_files: # For now just one mouse at a time thx
f = data_files[4]
h5f = pandas.HDFStore(data_dir + f, mode="r")['table']
phovect_cols = ['consonant','speaker','vowel','token']
h5f.phovect = h5f[phovect_cols]
mfcc_trials = np.ndarray((h5f.phovect.__len__(),20,201))
for i in range(0,h5f.phovect.__len__()):
    phovect = tuple(h5f.phovect.iloc[i])
    this_mfcc = mfccs[phovect_xdi[phovect]]
    # test = np.split(this_mfcc, this_mfcc.shape[1], axis=1) # Make 2d array a list of vectors
    mfcc_trials[i, 0:this_mfcc.shape[0],0:this_mfcc.shape[1]] = this_mfcc

#test = np.array(this_mfcc[0])
#for i in this_mfcc:


# Get responses
responses = h5f['response']

# Take out any nans & convert responses to 0 and 1
response_nans = np.isnan(responses)
keep_trials = response_nans[response_nans==False].index
responses = responses[keep_trials]
mfcc_trials = mfcc_trials[keep_trials,:,:]
responses[responses==1] = 0
responses[responses==3] = 1

# Reshape & Rescale
mfss_rs = mfcc_trials.reshape(len(responses), -1)
mfss_rs = preprocessing.scale(mfss_rs.astype(float))
mfss_rs_ind = preprocessing.scale(mfccs_np.astype(float))

# Split into training & test sets
split =  np.round(mfss_rs.shape[0]*.80).astype(int)
mfss_train = mfss_rs[0:split,:]
responses_train = responses[0:split]
mfss_test = mfss_rs[split+1:,:]
responses_test = responses[split+1:]


# Get sklernin on some trees
clf = RandomForestClassifier(n_estimators=500,n_jobs=6,max_features=128)
clf.fit(mfss_train,responses_train)

efc = ExtraTreesClassifier(n_estimators=1000,
                           n_jobs=6,
                           random_state=0,
                           max_features=128)
efc.fit(mfss_train,responses_train)

# Score the classifier
score = clf.score(mfss_test,responses_test)
score = efc.score(mfss_test,responses_test)

# Confusion matrix
correx = np.ndarray(len(mfccs_np))
correx[:80] = 0
correx[80:] = 1
predicts = clf.predict(mfccs_np)
metrics.confusion_matrix(correx,predicts)

# Plot contributions
prediction, bias, contributions = ti.predict(clf,mfss_rs_ind)
contrib_b = contributions[50,:,0]
contrib_g = contributions[50,:,1]
contrib_b_avg = np.mean(contrib_b,axis=0)
contrib_g_avg = np.mean(contrib_g,axis=0)

figure = plt.figure()
ax = plt.subplot(2,1,1)
ax.imshow(contrib_g_avg.reshape(20,201),origin='lower')
ax = plt.subplot(2,1,2)
plt.imshow(contrib_b.reshape(20,201),origin='lower')

figure = plt.figure()
ax = plt.subplot(4,1,1)
ax.imshow(mfss_rs_ind[50].reshape(20,201),origin='lower')
ax = plt.subplot(4,1,2)
ax.imshow(contrib_b.reshape(20,201),origin='lower')
ax = plt.subplot(4,1,3)
ax.imshow(contrib_g.reshape(20,201),origin='lower')
ax = plt.subplot(4,1,4)
ax.imshow(importances,origin='lower')

contrib_b_avg_shape = contrib_b_avg.reshape(20,201)

probs = clf.predict_proba(mfss_rs_ind)
importances = clf.feature_importances_.reshape(20,201)

# Plot mfcc
figure = plt.figure()
plt.imshow(mfss[10])
plt.imshow(importances,origin='lower')

# Export the data for plotting elsewhere
np.savetxt("/Users/Jonny/Documents/tempnpdat/contrib.csv",contrib_b,delimiter=",")
np.savetxt("/Users/Jonny/Documents/tempnpdat/contrib_avg_mfss.csv",contrib_b_avg_shape,delimiter=",")
np.savetxt("/Users/Jonny/Documents/tempnpdat/import_mfss.csv",importances,delimiter=",")
np.savetxt("/Users/Jonny/Documents/tempnpdat/probs.csv",probs,delimiter=",")
np.savetxt("/Users/Jonny/Documents/tempnpdat/mfcc.csv",np.rot90(mfccs[0]),delimiter=",")


# Plot that shit
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
xx,yy = np.meshgrid(np.arange(-1,1,0.2),
                    np.arange(-1,1,0.2))

xx,yy = np.meshgrid(np.arange(0, rs.shape[1] ),
                    np.arange(-1, 1, 0.02))


estimator_alpha = 1.0 / len(clf.estimators_)
cmap = plt.cm.RdYlBu
for tree in clf.estimators_:
    #for i in range(0,)
    #Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = tree.predict(yy)
    Z = Z.reshape(mfcc_trials.shape[1:2])
    cs = plt.contourf(range(0,200), range(0,13),np.rot90(mfcc_trials[1,:,:]), cmap=cmap)
    cs = plt.contourf(range(0, 13), range(0, 200), mfcc_trials[1, :, :], cmap=cmap)

plt.axis("tight")
plt.show()


Z = contributions[0,:,1].reshape(mfcc_trials.shape[1],mfcc_trials.shape[2])

# TODO: IF length of arrays different, alert me
# TODO: Fix '7012' and '7007' because they have bullshit phovects.





# TODO: remember to convert responses to 0 and 1. Make feature_arrays by matching tuple in data_arrays to phovect_ind
    # and indexing the mfccs, make response arrays by just taking the list, then use as input to classifier

############
# Hidden Markov Model
model = hmm.GaussianHMM()