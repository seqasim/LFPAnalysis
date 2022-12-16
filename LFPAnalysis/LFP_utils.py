import numpy as np
import re
 

def wm_ref():
    """
    Define a custom reference using the white matter electrodes. Choose the best 
    (most quiet) white matter reference on each electrode. 
    
    Then, manually subtract the reference from the LFP for each electrode and 
    return the new data. 

    Also rename the electrodes accordingly.
    """

    return wm_ref_data


def bipolar_ref():
    """
    Return the cathode list and anode list for mne to use for bipolar referencing.
    """

    # helper function to perform sort for bipolar electrodes:
    def num_sort(string):
        return list(map(int, re.findall(r'\d+', string)))[0]


    return cathode_list, anode_list


def match_elec_names(rec_names, pdf_names):
    """
    The electrode names in the pdf (used for localization) do not always match the electrode 
    names in the recording. This function tries matches the two.

    1. strip spaces from rec_names and put in lower case
    2. 
    """

def bad_electrode_detection(all_channels): 
    """
    Find outlier channels using a combination of kurtosis, variance, and standard deviation.
    
    https://www-sciencedirect-com.eresources.mssm.edu/science/article/pii/S016502701930278X
    
    Plot these channels for manual verification. 
    """


def IED_detection(channel, k=7): 
    """
    This function detects IEDs in the LFP signal.

    Method 1: Compute power of LFP signal in the [50-200] Hz band. Find z > 1. 
    (https://www.nature.com/articles/s41598-020-76138-7)
    Method 2: Bandpass filter in the [25-80] Hz band. Rectify. Find filtered envelope > 3. 
    (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6821283/)

    After each method, use k-means clustering to cluster IEDs by amplitude and timing. 

    Plot each cluster for visual inspection. 

    Remove outlying clusters manually until satisfied. 
    """

    pass


def remove_bad_data(mne_epoch_object, bad_channels, bad_epochs): 
    """
    Remove bad channels, and epochs from the mne Epoch object. 
    """

    pass

    
