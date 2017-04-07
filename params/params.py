class Speech_params(object):
    # Params are class attributes so we can use them without instantiation

    NAMES = ['Jonny','Ira','Anna','Dani','Theresa']
    CONS  = ['g','b']
    VOWS = ['I','o','a','ae','e','u']
    BASE_DIR = '/home/lab/github/SVN-ComboPack/ratrixSounds/phonemes/'
    MAP_BACK = {'g':1,'b':2,
           'Jonny':1,'Ira':2,'Anna':3,'Dani':4,'Theresa':5,
           'I':1,'o':2,'a':3,'ae':4,'e':5,'u':6}

    def init(self):
        pass

class Learn_params(object):
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
    N_JIT = 20
    JIT_AMT = 19 # In seconds or time bins

    # Number of models to train per run
    N_MODELS = 5

    # location of models when loading
    MODEL_DIR = "/mnt/data/speech_models/lin_conscat"
    N_STIM = 5
    CLASSES = ['1g','2g','3g','4g','5g','1b','2b','3b','4b','5b']
