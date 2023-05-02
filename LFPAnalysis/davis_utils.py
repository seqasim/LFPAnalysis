import numpy as np 
import pandas as pd

def UCD_check_edf_names(ch_names):

    #loop through channel names
    clean_names = []
    for ch in ch_names:
        if 'EEG' in ch:
            clean_names.append(ch.replace('EEG ','').replace('-REF','').lower())
        elif 'EEG' not in ch:
            clean_names.append(ch.lower())
        else:
            clean_names.append(ch.lower())
    if not clean_names:
        print('There is a problem with your channel naming format.')
    
    return clean_names


