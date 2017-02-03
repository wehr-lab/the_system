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
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from treeinterpreter import treeinterpreter as ti



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

MEL_PARAMS = {
    "n_fft"      : 480,
    "hop_length" : 240,
    "n_mels"     : 20,
    "fmin"       : 1000
}

PHOVECT_COLS = ['consonant','speaker','vowel','token']
NAMES = ['Jonny','Ira','Anna','Dani','Theresa']
CONS = ['g','b']
VOWS = ['I','o','a','ae','e','u']
#phoneme_dir = '/home/lab/github/SVN-ComboPack/ratrixSounds/phonemes/'
PHONEME_DIR = '/Users/Jonny/Github/SVN-ComboPack/ratrixSounds/phonemes/'

MAPBACK = {'g':1,'b':2,
           'Jonny':1,'Ira':2,'Anna':3,'Dani':4,'Theresa':5,
           'I':1,'o':2,'a':3,'ae':4,'e':5,'u':6}

DATA_DIR = '/Users/Jonny/Documents/fixSpeechData/'

PROP_TRAIN = 0.8 # Proportion of data to use as training data



###########################


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