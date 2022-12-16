import numpy as np
import re
 

def drop_irrelevant_channels(raw_data):
    """
    Drop the irrelevant channels from the raw data. 
    """

    return raw_data

def wm_ref(loc_data, bad_channels):
    """
    Define a custom reference using the white matter electrodes. (as in https://www.science.org/doi/10.1126/sciadv.abf4198)
    
    Identify all white matter electrodes (based on the electrode names), and make sure they are not bad electrodes (based on the bad channels list).

    1. iterate through each electrode, compute distance to all white matter electrodes 
    2. find 3 closest wm electrodes, compute amplitude (rms) 
    3. lowest amplitude electrode = wm reference 
    """

    # get the white matter electrodes and make sure they are note in the bad channel list
    if 'Manual Examination' in loc_data.keys():
        wm_elec_ix = [ind for ind, data in loc_data['Manual Examination'].str.lower().items() if data=='wm' and loc_data['label'][ind] not in bad_channels]
        oob_elec_ix = [ind for ind, data in loc_data['Manual Examination'].str.lower().items() if data=='oob']
    else: # this means we haven't doublechecked the electrode locations manually but trust the automatic locations
        wm_elec_ix = [ind for ind, data in loc_data['gm'].str.lower().items() if data=='white' and loc_data['label'][ind] not in bad_channels]
        oob_elec_ix = [ind for ind, data in loc_data['gm'].str.lower().items() if data=='unknown']
    
    # Make sure we don't include any microelectrodes here 
    wm_elec_ix = [x for x in wm_elec_ix if loc_data['label'][x][0]!='u']
    all_ix = loc_data.index.values
    gm_elec_ix = [x for x in all_ix if x not in wm_elec_ix and x not in oob_elec_ix]
    # Make sure we don't include any microelectrodes here 
    gm_elec_ix = [x for x in gm_elec_ix if loc_data['label'][x][0]!='u']


    cathode_list = []
    anode_list = []
    # reference is anode - cathode, so here wm is cathode

    for elec_ix in gm_elec_ix:
        # get the electrode location
        elec_loc = loc_data.loc[elec_ix, ['x', 'y', 'z']].values
        # compute the distance to all wm electrodes
        wm_elec_dist = np.sqrt(np.sum((loc_data.loc[wm_elec_ix, ['x', 'y', 'z']].values - elec_loc)**2, axis=1))
        # get the 3 closest wm electrodes
        wm_elec_ix_closest = wm_elec_ix[np.argsort(wm_elec_dist)[:3]]
        # get the amplitude of the 3 closest wm electrodes
        wm_elec_amp = np.sqrt(np.sum((loc_data.loc[wm_elec_ix_closest, ['x', 'y', 'z']].values - elec_loc)**2, axis=1))
        # get the index of the lowest amplitude electrode
        wm_elec_ix_lowest = wm_elec_ix_closest[np.argmin(wm_elec_amp)]
        # get the name of the lowest amplitude electrode
        wm_elec_name = loc_data.loc[wm_elec_ix_lowest, 'label']
        # get the electrode name
        elec_name = loc_data.loc[elec_ix, 'label']
        anode_list.append(elec_name)
        cathode_list.append(wm_elec_name)

    cathode_list = np.hstack(cathode_list)
    anode_list = np.hstack(anode_list)

    return anode_list, cathode_list  


def bipolar_ref(loc_data, bad_channels):
    """
    Return the cathode list and anode list for mne to use for bipolar referencing.
    """

    # helper function to perform sort for bipolar electrodes:
    def num_sort(string):
        return list(map(int, re.findall(r'\d+', string)))[0]


    # # identify the bundles for re-referencing:
    # loc_data['bundle'] = np.nan
    # loc_data['bundle'] = loc_data.apply(lambda x: ''.join(i for i in x.label if not i.isdigit()), axis=1)

    # cathode_list = [] 
    # anode_list = [] 
    # names = [] 
    # ref_category = [] 
    # # make a new elec_df 
    # elec_df_bipolar = [] 
    # for bundle in loc_data.bundle.unique():
        
    #     if bundle[0] == 'u':
    #         print('this is a microwire, pass')
    #         continue
                        
    #     # Isolate the electrodes in each bundle 
    #     bundle_df = loc_data[loc_data.bundle==bundle].sort_values(by='z', ignore_index=True)
        
    #     all_elecs = loc_data.label.tolist()
    #     # Sort them by number 
    #     all_elecs.sort(key=num_sort)
    #     # make sure these are not bad channels 
    #     all_elecs = [x for x in all_elecs if x not in bad_channels]
    #     # & (x.lower() not in MS007_data.info['bads']))
    #     # Set the cathodes and anodes 
    #     cath = all_elecs[1:]
    #     an = all_elecs[:-1]
    #     cathode_list.append(cath)
    #     anode_list.append(an)


    
    # cathode_list = np.hstack(cathode_list)
    # anode_list = np.hstack(anode_list)


    # return anode_list, cathode_list


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

def detect_bad_elecs(all_channels, loc_data): 
    """
    Find outlier channels using a combination of kurtosis, variance, and standard deviation. Also use the loc_data to find channels out of the brain
    
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

    
