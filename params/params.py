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