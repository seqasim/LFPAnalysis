import numpy as np
import re
import difflib 
from mne.preprocessing.bads import _find_outliers
from scipy.stats import kurtosis
import neurodsp

def wm_ref(mne_data, loc_data, bad_channels, unmatched_seeg=None):
    """
    Define a custom reference using the white matter electrodes. (as in https://www.science.org/doi/10.1126/sciadv.abf4198)
    
    Identify all white matter electrodes (based on the electrode names), and make sure they are not bad electrodes (based on the bad channels list).

    1. iterate through each electrode, compute distance to all white matter electrodes 
    2. find 3 closest wm electrodes, compute amplitude (rms) 
    3. lowest amplitude electrode = wm reference 

    Make sure it's the same hemisphere. If it can be on the same shaft that's great.    
    """

    # Drop the micros and unmatched seeg from here for now....
    drop_from_locs = []
    for ind, data in loc_data['label'].str.lower().items(): 
        if data in unmatched_seeg:
            drop_from_locs.append(ind)
        elif data[0] == 'u':
            drop_from_locs.append(ind)

    loc_data = loc_data.drop(index=drop_from_locs).reset_index(drop=True)

    # get the white matter electrodes and make sure they are note in the bad channel list
    if 'Manual Examination' in loc_data.keys():
        wm_elec_ix = [ind for ind, data in loc_data['Manual Examination'].str.lower().items() if data=='wm' and loc_data['label'][ind] not in bad_channels]
        oob_elec_ix = [ind for ind, data in loc_data['Manual Examination'].str.lower().items() if data=='oob']
    else: # this means we haven't doublechecked the electrode locations manually but trust the automatic locations
        wm_elec_ix = [ind for ind, data in loc_data['gm'].str.lower().items() if data=='white' and loc_data['label'][ind] not in bad_channels]
        oob_elec_ix = [ind for ind, data in loc_data['gm'].str.lower().items() if data=='unknown']


    all_ix = loc_data.index.values
    gm_elec_ix = np.array([x for x in all_ix if x not in wm_elec_ix and x not in oob_elec_ix])
    wm_elec_ix = np.array(wm_elec_ix)

    cathode_list = []
    anode_list = []
    drop_wm_channels = []
    # reference is anode - cathode, so here wm is cathode

    # NOTE: This loop is SLOW AF: is there a way to vectorize this for speed?
    for elec_ix in gm_elec_ix:
        # get the electrode location
        elec_loc = loc_data.loc[elec_ix, ['x', 'y', 'z']].values.astype(float)
        elec_name = loc_data.loc[elec_ix, 'label'].lower()
        # compute the distance to all wm electrodes
        wm_elec_dist = np.linalg.norm(loc_data.loc[wm_elec_ix, ['x', 'y', 'z']].values - elec_loc, axis=1)
        # get the 3 closest wm electrodes
        wm_elec_ix_closest = wm_elec_ix[np.argsort(wm_elec_dist)[:4]]
        # only keep the ones in the same hemisphere: 
        wm_elec_ix_closest = [x for x in wm_elec_ix_closest if loc_data.loc[x, 'label'].lower()[0]==elec_name[0]]
        # get the amplitude of the 3 closest wm electrodes
        wm_data = mne_data.copy().pick_channels(loc_data.loc[wm_elec_ix_closest, 'label'].str.lower().tolist())._data
        wm_elec_amp = wm_data.mean(axis=1)
        # get the index of the lowest amplitude electrode
        wm_elec_ix_lowest = wm_elec_ix_closest[np.argmin(wm_elec_amp)]
        # get the name of the lowest amplitude electrode
        wm_elec_name = loc_data.loc[wm_elec_ix_lowest, 'label'].lower()
        # get the electrode name
        anode_list.append(elec_name)
        cathode_list.append(wm_elec_name)
        
    # Also collect the wm electrodes that are not used for referencing and drop them later
    drop_wm_channels = [x for x in loc_data.loc[wm_elec_ix, 'label'].str.lower() if x not in cathode_list]
    oob_channels = loc_data.loc[oob_elec_ix, 'label'].str.lower().tolist()

    # cathode_list = np.hstack(cathode_list)
    # anode_list = np.hstack(anode_list)

    return anode_list, cathode_list, drop_wm_channels, oob_channels


def bipolar_ref(loc_data, bad_channels):
    """
    Return the cathode list and anode list for mne to use for bipolar referencing.

    TODO: Later - not a priority if white matter referencing is working.
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
    The electrode names read out of the edf file do not always match those 
    in the pdf (used for localization). This could be error on the side of the tech who input the labels, 
    or on the side of MNE reading the labels in. Usually there's a mixup between lowercase 'l' and capital 'I'. 

    This function matches the MNE channel names to those used in the localization. 

    params:
        mne_names: list of electrode names in the recording data (mne)
        loc_names: list of electrode names in the pdf, used for the localization
    """
    # strip spaces from mne_names and put in lower case
    mne_names = [x.replace(" ", "").lower() for x in mne_names]
    new_mne_names = mne_names.copy()

    # put loc_names in lower case
    loc_names = loc_names.str.lower()

    # Check which electrode names are in the loc but not the mne
    unmatched_names = list(set(loc_names) - set(mne_names))

    # seeg electrodes start with 'r' or 'l' - find the elecs in the mne names which are not in the localization data
    unmatched_seeg = [x for x in unmatched_names if x[0] in ['r', 'l']]

    # use string matching logic to try and determine if they are just misspelled (often i's and l's are mixed up)
    # (this is a bit of a hack, but should work for the most part)
    cutoff=0.8
    matched_elecs = []
    for elec in unmatched_seeg:
        # find the closest matches in each list. 
        match = difflib.get_close_matches(elec, mne_names, n=2, cutoff=cutoff)
        # if this fails, iteratively lower the cutoff until it works (to a point):
        while (len(match) == 0) & (cutoff >= 0.6):
            cutoff -= 0.05
            match = difflib.get_close_matches(elec, mne_names, n=2, cutoff=cutoff)
        if len(match) > 1: # pick the match with the correct hemisphere
            match = [x for x in match if x.startswith(elec[0])]
        if len(match) > 1: # if both are correct, pick the one with the correct #
            match = [x for x in match if x.endswith(elec[-1])]
        if len(match)>0:   
            # agree on one name: the localization name 
            new_mne_names[mne_names.index(match[0])] = elec
            matched_elecs.append(elec)
        else:
            print(f"Could not find a match for {elec}.")
    # drop the matched electrode from the unmatched lists
    unmatched_seeg = [i for i in unmatched_seeg if i not in matched_elecs]
    unmatched_names = [i for i in unmatched_names if i not in matched_elecs] # this should mostly be EEG and misc 

    return new_mne_names, unmatched_names, unmatched_seeg

def detect_bad_elecs(mne_struct, sEEG_mapping_dict): 
    """
    Find outlier channels using a combination of kurtosis, variance, and standard deviation. Also use the loc_data to find channels out of the brain
    
    https://www-sciencedirect-com.eresources.mssm.edu/science/article/pii/S016502701930278X
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7472198/
    https://www.biorxiv.org/content/10.1101/2021.05.14.444176v2.full.pdf

    
    Plot these channels for manual verification. 
    """

    # Get the data
    all_channels = mne_struct.pick_channels([*sEEG_mapping_dict])._data

    # Find bad channels
    kurt_chans = _find_outliers(kurtosis(all_channels, axis=1))
    var_chans = _find_outliers(np.var(all_channels, axis=1))
    std_chans = _find_outliers(np.std(all_channels, axis=1))
    kurt_chans = np.array([*sEEG_mapping_dict])[kurt_chans]
    var_chans = np.array([*sEEG_mapping_dict])[var_chans]
    std_chans = np.array([*sEEG_mapping_dict])[std_chans]

    # 

    return np.unique(kurt_chans.tolist() + var_chans.tolist() + std_chans.tolist()).tolist()

def detect_IEDs(channel, k=7): 
    """
    This function detects IEDs in the LFP signal automatically. Alternative to manual marking of each ied. 

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

    
