import numpy as np
import re
 

def drop_irrelevant_channels(raw_data):
    """
    Drop the irrelevant channels from the raw data. 
    """

    return raw_data

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


def match_elec_names(mne_names, loc_names):
    """
    The electrode names in the pdf (used for localization) do not always match the electrode 
    names in the recording. This function tries matches the two.
    params:
        mne_names: list of electrode names in the recording data (mne)
        loc_names: list of electrode names in the pdf, used for the localization
    """
    # strip spaces from mne_names and put in lower case
    mne_names = [x:x.replace(" ", "").lower() for x in mne_names]

    # put loc_names in lower case
    loc_names = loc_names.str.lower()

    # Check which electrodes are not shared (should just be micros, surface EEG, and trigger)
    unmatched_names = list(set(loc_names) ^ set(mne_names))

    # seeg electrodes start with 'r' or 'l' - find the elecs in the mne names which are not in the localization data
    unmatched_seeg = [x for x in unmatched_names if x[0] in ['r', 'l']]

    # use string matching logic to try and determine if they are just misspelled (often i's and l's are mixed up)
    # (this is a bit of a hack, but should work for the most part)
    for elec in unmatched_seeg:
        # find the closest match in the loc_names
        match = difflib.get_close_matches(elec, loc_names, n=1, cutoff=0.8)[0]
        # if this fails, iteratively lower the cutoff until it works (to a point):
        while (len(match) == 0) & (cutoff >= 0.5):
            cutoff -= 0.05
            match = difflib.get_close_matches(elec, loc_names, n=1, cutoff=cutoff)[0]
        if len(match) != 0:
            # replace the unmatched name with the matched name
            mne_names[mne_names.index(elec)] = match
            # drop the matched electrode from the unmatched lists
            unmatched_seeg.remove(elec)
            unmatched_names.remove(elec)
        else:
            print(f"Could not find a match for {elec}.")
    
    # how many seeg electrodes are left unmatched?
    print(f"{len(unmatched_seeg)} seeg electrodes were not matched to the localization data.")

    # which ones?
    print(unmatched_seeg)

    return unmatched_names, mne_names, loc_names

def detect_bad_elecs(all_channels): 
    """
    Find outlier channels using a combination of kurtosis, variance, and standard deviation.
    
    https://www-sciencedirect-com.eresources.mssm.edu/science/article/pii/S016502701930278X
    
    Plot these channels for manual verification. 
    """

    pass
    # return bad_channels

def detect_IEDs(channel, k=7): 
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

    # return bad_epochs


def remove_bad_data(mne_epoch_object, bad_channels, bad_epochs): 
    """
    Remove bad channels, and epochs from the mne Epoch object. 
    """

    pass

    
