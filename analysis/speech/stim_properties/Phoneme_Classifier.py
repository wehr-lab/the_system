__author__ = 'Jonny Saunders'
###############################
# Train a random forest to emulate a mouse's decision function on the MFCC's of each token
###############################

# Import Libraries
from python_speech_features import mfcc
import itertools
from scipy.io import wavfile

###########################
# Declare Parameters
# Empirically derived parameters - in general speech sounds have a narrower FFT window that other sounds
WIN_LENGTH = 0.005
WIN_STEP   =


###########################

# Declare scalar token number from phovect [consonant,speaker, vowel,token]
# Using stimmap 1 {Jonny, Ira, Anna, Dani, Theresa}
names = ['Jonny','Ira','Anna','Dani','Theresa']
cons = ['g','b']
vows = ['I','o','a','ae','e','u']
phoneme_dir = '/home/lab/github/SVN-ComboPack/ratrixSounds/phonemes/'

mapback = {'g':1,'b':2,
           'Jonny':1,'Ira':2,'Anna':3,'Dani':4,'Theresa':5,
           'I':1,'o':2,'a':3,'ae':4,'e':5,'u':6}

file_iterator = itertools.product(cons,names,vows,range(1,4))
counter = itertools.count()
phovect_idx = dict()
file_idx = dict()
file_loc = dict()
while True:
    try:
        file_components = file_iterator.next()
        if (file_components[1] == 'Jonny') & any(vow for vow in file_components[2] if vow in ['ae','e','u']):
            pass
        else:
            thisnum = counter.next()
            pho = file_components[0] + file_components[2]
            phovect_tup = (mapback[file_components[0]],mapback[file_components[1]],mapback[file_components[2]],file_components[3])
            phovect_idx[thisnum] = phovect_tup
            file_idx[thisnum] = file_components
            file_loc[thisnum] = phoneme_dir + file_components[1] + '/CV/' + pho + '/' + pho + str(file_components[3]) + '.wav'
    except StopIteration:
        break

# Load audio files & decompose into MFCC's
phonemes = dict()
phonemes_mfcc = dict()
for i, loc in file_loc:
    fs, phonemes[i] = wavfile.read(loc)






# Audio Feature Extraction

