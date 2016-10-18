__author__ = 'Jonny <jsaunder@uoregon.edu>'
import os
import re
import pandas
import numpy as np


def csv_to_h5(dir):
    """
    Convert all csv files in a directory to hdf5
    """
    os.chdir(dir)
    file_list = [f for f in os.listdir('.') if f.endswith('.csv')]
    for f in file_list:
        csv_dat = pandas.read_csv(f)
        csv_dat = csv_dat.replace("NaN",np.nan) # In case we don't correctly detect the NaNs
        csv_dat.to_hdf(re.sub(r'.csv','.h5',f),'table')

    print("Successfully made {} h5 files".format(len(file_list)))

