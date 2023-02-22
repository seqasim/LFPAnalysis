import numpy as np
import re
import difflib 
from mne.preprocessing.bads import _find_outliers
from scipy.stats import kurtosis, zscore
import neurodsp
import mne
from glob import glob
from LFPAnalysis import nlx_utils, lfp_preprocess_utils
import pandas as pd
from mne.filter import next_fast_len
from scipy.signal import hilbert, find_peaks, peak_widths

def mean_baseline_time(data, baseline, mode='zscore'): 
    
    """
    Meant to mimic the mne baseline for time-series but when the specific baseline period might change across trials, as 
    mne doesn't allow baseline period to vary. 

    Parameters
    ----------
    data : 2d numpy array
        original data
    baseline : 2d numpy array
        baseline data 
    mode : str
        choice of baseline mode
    Returns
    -------
    baseline_corrected : 2d numpy array
        baselined data 
    """
    
    baseline_mean =  baseline.mean(axis=-1)
    m = np.expand_dims(baseline_mean, axis=2)
    baseline_std = baseline.std(axis=-1)
    std = np.expand_dims(baseline_std, axis=2)


    if mode == 'mean':
        baseline_corrected = data - m
    elif mode == 'ratio':
        baseline_corrected = data / m
    elif mode == 'logratio':
        baseline_corrected = np.log10(data / m)
    elif mode == 'percent':
        baseline_corrected = (data - m) / m 
    elif mode == 'zscore':
        baseline_corrected = (data - m) / std 
    elif mode == 'zlogratio':
        baseline_corrected = np.log10(data / m) / std

    return baseline_corrected 

def zscore_TFR_average(data, baseline, mode='zscore'): 
    
    """
    Meant to mimic the mne baseline (specifically just the zscore for now) 
    for TFR but when the specific baseline period might change across trials. 

    This presumes you're using trial-averaged data (check dimensions)
    
    Parameters
    ----------
    data : 2d numpy array
        original data
    baseline : 2d numpy array
        baseline data 
    mode : str
        choice of baseline mode
    Returns
    -------
    baseline_corrected : 2d numpy array
        baselined data 
    """
    

    m = baseline.mean(axis=-1)
    m = np.expand_dims(m, axis=2)
    m = np.repeat(m,  data.shape[-1], axis=2)
    
    std = baseline.std(axis=-1)
    std = np.expand_dims(std, axis=2)
    std = np.repeat(std,  data.shape[-1], axis=2)

    if mode == 'mean':
        baseline_corrected = data - m
    elif mode == 'ratio':
        baseline_corrected = data / m
    elif mode == 'logratio':
        baseline_corrected = np.log10(data / m)
    elif mode == 'percent':
        baseline_corrected = (data - m) / m 
    elif mode == 'zscore':
        baseline_corrected = (data - m) / std 
    elif mode == 'zlogratio':
        baseline_corrected = np.log10(data / m) / std
    
    return baseline_corrected 

def zscore_TFR_across_trials(data, baseline, mode='zscore'): 
    
    """
    Meant to mimic the mne baseline (specifically just the zscore for now) 
    for TFR but when the specific baseline period might change across trials. 

    This presumes you're using trial-level data (check dimensions)
    
    Parameters
    ----------
    data : 2d numpy array
        original data
    baseline : 2d numpy array
        baseline data 
    mode : str
        choice of baseline mode
    Returns
    -------
    baseline_corrected : 2d numpy array
        baselined data 
    """
    
    # Create an array of the mean and standard deviation of the power values across the session
    # 1. Compute the mean for every electrode, at every frequency 
    m = np.mean(np.mean(baseline, axis=3) ,axis=0)
    # 2. Expand the array
    m = np.expand_dims(np.expand_dims(m, axis=0),axis=3)
    # 3. Copy the data to every event and time-point
    m = np.repeat(np.repeat(m, data.shape[0],axis=0), 
                  data.shape[-1],axis=3)

    std = np.std(np.mean(baseline, axis=3),axis=0)
    std = np.expand_dims(np.expand_dims(std, axis=0),axis=3)
    std = np.repeat(np.repeat(std, data.shape[0], axis=0), 
                   data.shape[-1],axis=3)

    if mode == 'mean':
        baseline_corrected = data - m
    elif mode == 'ratio':
        baseline_corrected = data / m
    elif mode == 'logratio':
        baseline_corrected = np.log10(data / m)
    elif mode == 'percent':
        baseline_corrected = (data - m) / m 
    elif mode == 'zscore':
        baseline_corrected = (data - m) / std 
    elif mode == 'zlogratio':
        baseline_corrected = np.log10(data / m) / std
    
    return baseline_corrected 

def wm_ref(mne_data, elec_data, bad_channels, unmatched_seeg=None, site=None, average=False):
    """
    Define a custom reference using the white matter electrodes. Originated here: https://doi.org/10.1016/j.neuroimage.2015.02.031

    (as in https://www.science.org/doi/10.1126/sciadv.abf4198)
    
    Identify all white matter electrodes (based on the electrode names), and make sure they are not bad electrodes (based on the bad channels list).

    1. iterate through each electrode, compute distance to all white matter electrodes 
    2. find 3 closest wm electrodes, compute amplitude (rms) 
    3. lowest amplitude electrode = wm reference 

    Make sure it's the same hemisphere. 
    
    TODO: implement average reference option, whereby the mean activity across all white matter electrodes is used as a reference [separate per hemi]... 
    see: https://www.sciencedirect.com/science/article/pii/S1053811922005559#bib0349

    TODO: this is SLOW; any vectorization to speed it up or parallelization?

    Parameters
    ----------
    mne_data : mne object
        non-referenced data stored in an MNE object 
    elec_data : pandas df 
        dataframe containing the electrode localization information
    bad_channels : list 
        bad channels 
    unmatched_seeg : list 
        list of channels that were not in the edf file 
    site : str
        hospital where the recording took place 
    average : bool 
        should we construct an average white matter reference instead of a default? 

    Returns
    -------
    anode_list : list 
        list of channels to subtract from
    cathode_list : list 
        list of channels to subtract
    drop_wm_channels : list 
        list of white matter channels which were not used for reference and now serve no purpose 

    """

    if site == 'MSSM': 
        # Drop the micros and unmatched seeg from here for now....
        drop_from_locs = []
        for ind, data in elec_data['label'].str.lower().items(): 
            if data in unmatched_seeg:
                drop_from_locs.append(ind)
            elif data[0] == 'u':
                drop_from_locs.append(ind)

        elec_data = elec_data.drop(index=drop_from_locs).reset_index(drop=True)

        # get the white matter electrodes and make sure they are note in the bad channel list
        wm_elec_ix_manual = [] 
        wm_elec_ix_auto = []
        if 'Manual Examination' in elec_data.keys():
            wm_elec_ix_manual = wm_elec_ix_manual + [ind for ind, data in elec_data['Manual Examination'].str.lower().items() if data=='wm' and elec_data['label'].str.lower()[ind] not in bad_channels]
            oob_elec_ix = [ind for ind, data in elec_data['Manual Examination'].str.lower().items() if data=='oob']
        else: # this means we haven't doublechecked the electrode locations manually but trust the automatic locations
            wm_elec_ix_auto = wm_elec_ix_auto + [ind for ind, data in elec_data['gm'].str.lower().items() if data=='white' and elec_data['label'].str.lower()[ind] not in bad_channels]
            oob_elec_ix = [ind for ind, data in elec_data['gm'].str.lower().items() if data=='unknown']

        wm_elec_ix = np.unique(wm_elec_ix_manual + wm_elec_ix_auto)
        all_ix = elec_data.index.values
        gm_elec_ix = np.array([x for x in all_ix if x not in wm_elec_ix and x not in oob_elec_ix])

        cathode_list = []
        anode_list = []
        drop_wm_channels = []
        # reference is anode - cathode, so here wm is cathode

        # NOTE: This loop is SLOW AF: is there a way to vectorize this for speed?
        for elec_ix in gm_elec_ix:
            # get the electrode location
            elec_loc = elec_data.loc[elec_ix, ['x', 'y', 'z']].values.astype(float)
            elec_name = elec_data.loc[elec_ix, 'label'].lower()
            # compute the distance to all wm electrodes
            wm_elec_dist = np.linalg.norm(elec_data.loc[wm_elec_ix, ['x', 'y', 'z']].values - elec_loc, axis=1)
            # get the 3 closest wm electrodes
            wm_elec_ix_closest = wm_elec_ix[np.argsort(wm_elec_dist)[:4]]
            # only keep the ones in the same hemisphere: 
            wm_elec_ix_closest = [x for x in wm_elec_ix_closest if elec_data.loc[x, 'label'].lower()[0]==elec_name[0]]
            # get the variance of the 3 closest wm electrodes
            wm_data = mne_data.copy().pick_channels(elec_data.loc[wm_elec_ix_closest, 'label'].str.lower().tolist())._data
            wm_elec_var = wm_data.var(axis=1)
            # get the index of the lowest variance electrode
            wm_elec_ix_lowest = wm_elec_ix_closest[np.argmin(wm_elec_var)]
            # get the name of the lowest amplitude electrode
            wm_elec_name = elec_data.loc[wm_elec_ix_lowest, 'label'].lower()
            # get the electrode name
            anode_list.append(elec_name)
            cathode_list.append(wm_elec_name)
            
        # Also collect the wm electrodes that are not used for referencing and drop them later
        drop_wm_channels = [x for x in elec_data.loc[wm_elec_ix, 'label'].str.lower() if x not in cathode_list]
        oob_channels = elec_data.loc[oob_elec_ix, 'label'].str.lower().tolist()

        # cathode_list = np.hstack(cathode_list)
        # anode_list = np.hstack(anode_list)

        return anode_list, cathode_list, drop_wm_channels, oob_channels

    elif site == 'UI':
        wm_elec_ix = [ind for ind, data in elec_data['region_1'].str.lower().items() if 'white' in data and elec_data['Channel'][ind] not in mne_data.info['bads']]
        all_ix = elec_data.index.values
        gm_elec_ix = np.array([x for x in all_ix if x not in wm_elec_ix])
        wm_elec_ix = np.array(wm_elec_ix)

        cathode_list = []
        anode_list = []
        drop_wm_channels = []
        # reference is anode - cathode, so here wm is cathode

        # NOTE: This loop is SLOW AF: is there a way to vectorize this for speed?
        for elec_ix in gm_elec_ix:
            # get the electrode location
            elec_loc = elec_data.loc[elec_ix, ['mni_x', 'mni_y', 'mni_z']].values.astype(float)
            elec_name = elec_data.loc[elec_ix, 'Channel'].lower()
            # compute the distance to all wm electrodes
            wm_elec_dist = np.linalg.norm(elec_data.loc[wm_elec_ix, ['mni_x', 'mni_y', 'mni_z']].values - elec_loc, axis=1)
            # get the 3 closest wm electrodes
            wm_elec_ix_closest = wm_elec_ix[np.argsort(wm_elec_dist)[:4]]
            # only keep the ones in the same hemisphere: 
            wm_elec_ix_closest = [x for x in wm_elec_ix_closest if elec_data.loc[x, 'label'].lower()[0]==elec_data.loc[elec_ix, 'label'].lower()[0]]
            # get the variance of the 3 closest wm electrodes
            wm_data = mne_data.copy().pick_channels(elec_data.loc[wm_elec_ix_closest, 'Channel'].str.lower().tolist())._data
            wm_elec_var = wm_data.var(axis=1)
            # get the index of the lowest variance electrode
            wm_elec_ix_lowest = wm_elec_ix_closest[np.argmin(wm_elec_var)]
            # get the name of the lowest amplitude electrode
            wm_elec_name = elec_data.loc[wm_elec_ix_lowest, 'Channel'].lower()
            # get the electrode name
            anode_list.append(elec_name)
            cathode_list.append(wm_elec_name)

        # Also collect the wm electrodes that are not used for referencing and drop them later
        drop_wm_channels = [x for x in elec_data.loc[wm_elec_ix, 'Channel'].str.lower() if x not in cathode_list]

        return anode_list, cathode_list, drop_wm_channels


def laplacian_ref(mne_data, elec_data, bad_channels, unmatched_seeg=None, site=None):
    """
    Return the cathode list and anode list for mne to use for laplacian referencing.

    In this case, the cathode is the average of the surrounding electrodes. If an edge electrode, it's just bipolar. 

    From here: https://doi.org/10.1016/j.neuroimage.2018.08.020

    """

    pass

def bipolar_ref(elec_data, bad_channels, unmatched_seeg=None, site=None):
    """
    Return the cathode list and anode list for mne to use for bipolar referencing.

    Parameters
    ----------
    elec_data : pandas df 
        dataframe containing the electrode localization information
    bad_channels : list 
        bad channels 
    unmatched_seeg : list 
        list of channels that were not in the edf file 
    site : str
        hospital where the recording took place 

    Returns
    -------
    anode_list : list 
        list of channels to subtract from
    cathode_list : list 
        list of channels to subtract
    """

    # helper function to perform sort for bipolar electrodes:
    def num_sort(string):
        return list(map(int, re.findall(r'\d+', string)))[0]

    cathode_list = [] 
    anode_list = [] 

    if site=='MSSM':

        for bundle in elec_data.bundle.unique():
            if bundle[0] == 'u':
                print('this is a microwire, pass')
                continue         
            # Isolate the electrodes in each bundle 
            bundle_df = elec_data[elec_data.bundle==bundle].sort_values(by='z', ignore_index=True)
            all_elecs = elec_data.label.tolist()
            # Sort them by number 
            all_elecs.sort(key=num_sort)
            # make sure these are not bad channels 
            all_elecs = [x for x in all_elecs if x not in bad_channels]
            # Set the cathodes and anodes 
            cath = all_elecs[1:]
            an = all_elecs[:-1]
            cathode_list = cathode_list + cath
            anode_list = anode_list + an

    elif site=='UI':

        for bundle in elec_data.bundle.unique():
            # Isolate the electrodes in each bundle 
            bundle_df = elec_data[elec_data.bundle==bundle].sort_values(by='contact', ignore_index=True)
            all_elecs = bundle_df.Channel.tolist()
            # make sure these are not bad channels 
            all_elecs = [x for x in all_elecs if x not in bad_channels]
            # Set the cathodes and anodes 
            cath = all_elecs[1:]
            an = all_elecs[:-1]
            cathode_list = cathode_list + cath
            anode_list = anode_list + an

    return anode_list, cathode_list


def match_elec_names(mne_names, loc_names):
    """
    The electrode names read out of the edf file do not always match those 
    in the pdf (used for localization). This could be error on the side of the tech who input the labels, 
    or on the side of MNE reading the labels in. Usually there's a mixup between lowercase 'l' and capital 'I'. 

    This function matches the MNE channel names to those used in the localization. 

    Parameters
    ----------
    mne_names : list
        list of electrode names in the recording data (mne)
    loc_names : list 
        list of electrode names in the pdf, used for the localization

    Returns
    -------
    new_mne_names : list 
        revised mne names merged across sources 
    unmatched_names : list 
        names that do not match (mostly scalp EEG and misc)
    unmatched_seeg : list
        sEEG channels that do not match (should be rare)
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

def detect_bad_elecs(mne_data, sEEG_mapping_dict): 
    """
    Find outlier channels using a combination of kurtosis, variance, and standard deviation. Also use the elec_data to find channels out of the brain
    
    https://www-sciencedirect-com.eresources.mssm.edu/science/article/pii/S016502701930278X
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7472198/
    https://www.biorxiv.org/content/10.1101/2021.05.14.444176v2.full.pdf

    
    Plot these channels for manual verification. 

    Parameters
    ----------
    mne_data : mne object 
        mne data to check for bad channels 
    sEEG_mapping_dict : dict 
        dict of sEEG channels 

    Returns
    -------
    bad_channels : list 
        list of bad channels 
    """

    # Get the data
    all_channels = mne_data.pick_channels([*sEEG_mapping_dict])._data

    # Find bad channels
    kurt_chans = _find_outliers(kurtosis(all_channels, axis=1))
    var_chans = _find_outliers(np.var(all_channels, axis=1))
    std_chans = _find_outliers(np.std(all_channels, axis=1))
    kurt_chans = np.array([*sEEG_mapping_dict])[kurt_chans]
    var_chans = np.array([*sEEG_mapping_dict])[var_chans]
    std_chans = np.array([*sEEG_mapping_dict])[std_chans]

    bad_channels = np.unique(kurt_chans.tolist() + var_chans.tolist() + std_chans.tolist()).tolist()
    # 
    return bad_channels

def detect_IEDs(mne_data, peak_thresh=5, closeness_thresh=0.25, width_thresh=0.2): 
    """
    This function detects IEDs in the LFP signal automatically. Alternative to manual marking of each ied. 

    Method 1: 
    1. Bandpass filter in the [25-80] Hz band. 
    2. Rectify. 
    3. Find filtered envelope > 3. 
    4. Eliminate events with peaks with unfiltered envelope < 3. 
    5. Eliminate close IEDs (peaks within 500 ms). 
    6. Eliminate IEDs that are not present on at least 4 electrodes. 
    (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6821283/)

    Parameters
    ----------
    mne_data : mne object 
        mne data to check for bad channels 
    peak_thresh : float 
        the peak threshold in amplitude 
    closeness_thresh : float 
        the closeness threshold in time
    width_thresh : float 
        the width threshold for IEDs 

    Returns
    -------
    IED_samps_dict : dict 
        dict with every IED index  

    """

    # What type of data is this? Continuous or epoched? 
    if type(mne_data) == mne.epochs.Epochs:
        data_type = 'epoch'
        n_times = mne_data._data.shape[-1]
    elif type(mne_data) == mne.io.fif.raw.Raw: 
        data_type = 'continuous'
        n_times = mne_data._data.shape[1]
    else: 
        data_type = 'continuous'
        n_times = mne_data._data.shape[1]       

    sr = mne_data.info['sfreq']
    min_width = width_thresh * sr
    across_chan_threshold_samps = closeness_thresh * sr # This sets a threshold for detecting cross-channel IEDs 

    # filter data in beta-gamma band
    filtered_data = mne_data.copy().filter(25, 80, n_jobs=-1)

    n_fft = next_fast_len(n_times)

    # Hilbert bandpass amplitude 
    filtered_data = filtered_data.apply_hilbert(envelope=True, n_fft=n_fft, n_jobs=-1)

    # Rectify: 
    filtered_data._data[filtered_data._data<0] = 0

    # Zscore
    filtered_data.apply_function(lambda x: zscore(x, axis=-1))
    IED_samps_dict = {f'{x}':np.nan for x in mne_data.ch_names}
    IED_sec_dict = {f'{x}':np.nan for x in mne_data.ch_names}

    if data_type == 'continuous':
        for ch_ in filtered_data.ch_names:
            sig = filtered_data.get_data(picks=[ch_])[0, :]

            # Find peaks 
            IED_samps, _ = find_peaks(sig, height=peak_thresh, prominence=2, distance=closeness_thresh * sr)

            IED_samps_dict[ch_] = IED_samps 

        # aggregate all IEDs
        all_IEDs = np.sort(np.concatenate(list(IED_samps_dict.values())).ravel())

        # Remove lame IEDs 
        for ch_ in filtered_data.ch_names:
            sig = filtered_data.get_data(picks=[ch_])[0, :]
            # 1. Too wide  
            # Whick IEDs are longer than 200 ms?
            widths = peak_widths(sig, IED_samps_dict[ch_], rel_height=0.75)
            wide_IEDs = np.where(widths[0] > min_width)[0]
            # 2. Too small 
            # Which IEDs are below 3 in z-scored unfiltered signal? 
            small_IEDs = np.where(zscore(mne_data.get_data(picks=[ch_]), axis=-1)[0, IED_samps_dict[ch_]] < 3)[0]
            local_IEDs = [] 
            # 3. Too local 
            # Which IEDs are not present on enough electrodes? 
            # Logic - aggregate IEDs across all channels as a reference point 
            # Check each channel's IED across aggregate to find ones that are close in time (but are<500 ms so can't be same channel)
            for IED_ix, indvid_IED in enumerate(IED_samps_dict[ch_]): 
                # compute the time (in samples) to all IEDS if the 5 closest aren't all within 100 ms, then reject
                diff_with_all_IEDs = np.sort(np.abs(indvid_IED - all_IEDs))[0:5]
                if any(diff_with_all_IEDs>=across_chan_threshold_samps): 
                    local_IEDs.append(IED_ix)
                    # print(diff_with_all_IEDs)
            local_IEDs = np.array(local_IEDs).astype(int)   
            elim_IEDs = np.unique(np.hstack([small_IEDs, wide_IEDs, local_IEDs]))
            revised_IED_samps = np.delete(IED_samps_dict[ch_], elim_IEDs)
            IED_s = (revised_IED_samps / sr)
            IED_sec_dict[ch_] = IED_s   
          
        return IED_sec_dict
    elif data_type == 'epoch':
        # Detect the IEDs in every event in epoch time
        for ch_ in filtered_data.ch_names:
            sig = filtered_data.get_data(picks=[ch_])[:,0,:]
            IED_dict = {x:np.nan for x in np.arange(sig.shape[0])}
            for event in np.arange(sig.shape[0]):
                IED_samps, _ = find_peaks(sig[event, :], height=peak_thresh, prominence=2, distance=closeness_thresh * sr)
                # IED_s = (IED_samps / sr)
                if len(IED_samps) == 0: 
                    IED_samps = np.array([np.nan])
                    # IED_s = np.nan
                IED_dict[event] = IED_samps
            IED_samps_dict[ch_] = IED_dict
        # aggregate all IEDs
        all_IEDs = np.sort(np.concatenate([list(x.values())[0] for x in list(IED_samps_dict.values())]).ravel())
        for ch_ in filtered_data.ch_names:
            sig = filtered_data.get_data(picks=[ch_])[:,0,:]
            for event in np.arange(sig.shape[0]):
                if all(~np.isnan(IED_samps_dict[ch_][event])): # Make sure there are IEDs here to begin with during this event 
                    widths = peak_widths(sig[event, :], IED_samps_dict[ch_][event], rel_height=0.75)
                    wide_IEDs = np.where(widths[0] > min_width)[0]
                    small_IEDs = np.where(zscore(mne_data.get_data(picks=[ch_]), axis=-1)[event, 0, IED_samps_dict[ch_][event]] < 3)[0]
                    local_IEDs = [] 
                    for IED_ix, indvid_IED in enumerate(IED_samps_dict[ch_][event]): 
                        # compute the time (in samples) to all IEDS if the 5 closest aren't all within 100 ms, then reject
                        diff_with_all_IEDs = np.sort(np.abs(indvid_IED - all_IEDs))[0:5]
                        if any(diff_with_all_IEDs>=across_chan_threshold_samps): 
                            local_IEDs.append(IED_ix)
                    local_IEDs = np.array(local_IEDs).astype(int)
                    elim_IEDs = np.unique(np.hstack([small_IEDs, wide_IEDs, local_IEDs]))
                    revised_IED_samps = np.delete(IED_samps_dict[ch_][event], elim_IEDs)
                    IED_samps_dict[ch_][event] = revised_IED_samps

        return IED_samps_dict

# Below are code that condense the Jupyter notebooks for pre-processing into individual functions. 

def make_mne(load_path=None, elec_data=None, format='edf', site='MSSM', overwrite=True, **kwargs):
    """
    Make a mne object from the data and electrode files, and save out the photodiode. 
    Following this step, you can indicate bad electrodes manually.
    
    Parameters
    ----------
    load_path : str
        path to the neural data
    elec_data : pandas df 
        dataframe with all the electrode localization information
    format : str 
        how was this data collected? options: ['edf', 'nlx]
    site: str
        where was the data collected? options: ['UI', 'MSSM'].
        TODO: add site specificity for UC Davis
    overwrite: bool 
        whether to overwrite existing data for this person if it exists 
    kwargs: dict
        dictionary containing lists of different types of channel names 

    Returns
    -------
    mne_data : mne object 
        mne object
    """

    # OPTIONAL: Set specific channel names that you might need: 
    eeg_names = kwargs['eeg_names']
    resp_names = kwargs['resp_names']
    ekg_names = kwargs['ekg_names']
    photodiode_name = kwargs['photodiode_name']
    seeg_names = kwargs['seeg_names']

    if site == 'MSSM':
        if not eeg_names: # If no input, assume the standard EEG montage at MSSM
            eeg_names = ['fp1', 'f7', 't3', 't5', 'o1', 'f3', 'c3', 'p3', 'fp2', 'f8', 't4', 't6', 'o2', 'f4', 'c4', 'p4', 'fz', 'cz', 'pz']
    
    # 1) load the data:
    if format=='edf':
        # MAKE SURE ALL THE EDF CHANNELS HAVE THE SAME SR! See: https://github.com/mne-tools/mne-python/issues/10635

        if not photodiode_name: 
            photodiode_name = 'dc1'
        # EDF data always comes from MSSM AFAIK. Modify this if that changes.

        # This is a big block of data. Have to load first, then split out the sEEG and photodiode downstream. 
        edf_file = glob(f'{load_path}/*.edf')[0]
        mne_data = mne.io.read_raw_edf(edf_file, preload=True)
        
        # The electrode names read out of the edf file do not always match those 
        # in the pdf (used for localization). This could be error on the side of the tech who input the labels, 
        # or on the side of MNE reading the labels in. Usually there's a mixup between lowercase 'l' and capital 'I'.
        
        # Sometimes, there's electrodes on the pdf that are NOT in the MNE data structure... let's identify those as well. 
        new_mne_names, _, _ = match_elec_names(mne_data.ch_names, elec_data.label)
        # Rename the mne data according to the localization data
        new_name_dict = {x:y for (x,y) in zip(mne_data.ch_names, new_mne_names)}
        mne_data.rename_channels(new_name_dict)

        if not seeg_names:
            seeg_names = [i for i in mne_data.ch_names if ((i.startswith('l')) | (i.startswith('r')))]
        sEEG_mapping_dict = {f'{x}':'seeg' for x in seeg_names}

        mne_data.set_channel_types(sEEG_mapping_dict)

        mne_data.info['line_freq'] = 60
        # Notch out 60 Hz noise and harmonics 
        mne_data.notch_filter(freqs=(60, 120, 180, 240))

        # Save out the photodiode channel separately
        mne_data.save(f'{load_path}/photodiode.fif', picks=photodiode_name, overwrite=overwrite)

        # drop EEG and EKG channels
        drop_chans = list(set(mne_data.ch_names)^set(seeg_names))
        mne_data.drop_channels(drop_chans)

        bads = detect_bad_elecs(mne_data, sEEG_mapping_dict)
        mne_data.info['bads'] = bads

    elif format =='nlx': 
        # This is a pre-split data. Have to specifically load the sEEG and photodiode individually.
        signals = [] 
        srs = [] 
        ch_name = [] 
        ch_type = []
        if site == 'MSSM': 
            # per Shawn, MSSM data seems to sometime have a "_0000.ncs" to "_9999.ncs" appended to the end of real data
            pattern = re.compile(r"_\d{4}\.ncs")  # regex pattern to match "_0000.ncs" to "_9999.ncs"
            ncs_files = [x for x in glob(f'{load_path}/*.ncs') if re.search(pattern, x)]
            # just in case this changes in the future: 
            if len(ncs_files) == 0: 
                ncs_files = glob(f'{load_path}/*.ncs')
                if not seeg_names:
                    seeg_names = [x.split('/')[-1].replace('.ncs','') for x in glob(f'{load_path}/[R,L]*.ncs')]
            else:
                if not seeg_names:
                    seeg_names = [x.split('/')[-1].replace('.ncs','').split('_')[0] for x in glob(f'{load_path}/[R,L]*.ncs') if re.search(pattern, x)]
        elif site == 'UI':
            # here, the filenames are not informative. We have to get subject-specific information from the experimenter
            ncs_files = glob(f'{load_path}/LFP*.ncs')
        if not seeg_names: 
            raise ValueError('no seeg channels specified')
        else:
            sEEG_mapping_dict = {f'{x}':'seeg' for x in seeg_names}

        for chan_path in ncs_files:
            chan_name = chan_path.split('/')[-1].replace('.ncs','')
            # strip the file type off the end if needed 
            if '_' in chan_name:
                chan_name = chan_name.split('_')[0]
            try:
                fdata = nlx_utils.load_ncs(chan_path)
            except IndexError: 
                print(f'No data in channel {chan_name}')
                continue
            #  surface eeg
            if chan_name.lower() in eeg_names:
                ch_type.append('eeg')
            elif chan_name.lower() in resp_names:
                ch_type.append('bio')
            elif chan_name.lower() in ekg_names: 
                ch_type.append('ecg')  
            elif chan_name.lower() in seeg_names:
                ch_type.append('seeg')  
            else:
                print(f'Unidentified data type in {chan_name}')
                continue
        
            signals.append(fdata['data'])
            srs.append(fdata['sampling_rate'])
            ch_name.append(chan_name)

        if np.unique(srs).shape[0] == 1:
            # all the sampling rates match:
            info = mne.create_info(ch_name, np.unique(srs), ch_type)
            mne_data = mne.io.RawArray(signals, info)
        else:
            ## Now we have to account for differing sampling rates. This will only really happen in the case of data where ANALOGUE channels 
            ## are recorded at a much higher sampling rate, or with micro channels. Find the lowest sample rate, and downsample everything to that.
            ## I generally don't like this but it should be OK. Make sure that you identify synchronization times AFTER downsampling the analogue channel, and not before:
            ## https://gist.github.com/larsoner/01642cb3789992fbca59

            target_sr = np.min(srs)
            mne_data_resampled = []

            for sr in np.unique(srs):
                ch_ix = np.where(srs==sr)[0].astype(int)
                info = mne.create_info([x for ix, x in enumerate(ch_name) if ix in ch_ix], sr, [x for ix, x in enumerate(ch_type) if ix in ch_ix])
                mne_data_temp = mne.io.RawArray([x for ix, x in enumerate(signals) if ix in ch_ix], info)
                if sr != target_sr:
                    # resample down to one sample rate 
                    mne_data_temp.resample(sfreq=target_sr)
                    mne_data_resampled.append(mne_data_temp)
                else: 
                    mne_data = mne_data_temp

            mne_data.add_channels(mne_data_resampled)

            mne_data.set_channel_types(sEEG_mapping_dict)

            mne_data.info['line_freq'] = 60
            # Notch out 60 Hz noise and harmonics 
            mne_data.notch_filter(freqs=(60, 120, 180, 240))

            # Save out the photodiode channel separately
            if not photodiode_name:
                raise ValueError('no photodiode channel specified')
            else:
                mne_data.save(f'{load_path}/photodiode.fif', picks=photodiode_name, overwrite=overwrite)

            # Save out the respiration channels separately
            if resp_names:
                mne_data.save(f'{load_path}/respiration_data.fif', picks=resp_names, overwrite=overwrite)
            
            # Save out the EEG channels separately
            if eeg_names: 
                mne_data.save(f'{load_path}/scalp_eeg_data.fif', picks=eeg_names, overwrite=overwrite)

            # Save out the EEG channels separately
            if ekg_names:
                mne_data.save(f'{load_path}/ekg_data.fif', picks=ekg_names, overwrite=overwrite)

            drop_chans = list(set(mne_data.ch_names)^set(seeg_names))
            mne_data.drop_channels(drop_chans)

            bads = detect_bad_elecs(mne_data, sEEG_mapping_dict)
            mne_data.info['bads'] = bads

            mne_data.save(f'{load_path}/lfp_data.fif', picks=seeg_names, overwrite=overwrite)

    return mne_data


def ref_mne(mne_data=None, elec_data=None, method='wm', site='MSSM'):
    """
    Following this step, you can indicate IEDs manually.

    Parameters
    ----------
    mne_data : mne object 
        mne object
    elec_data : pandas df 
        dataframe with all the electrode localization information
    method : str 
        how should we reference the data ['wm', 'bipolar']
    site : str 
        where was this data collected? Options: ['MSSM', 'UI', 'Davis']

    Returns
    -------
    mne_data_reref : mne object 
        mne object with re-referenced data
    """

    # Sometimes, there's electrodes on the pdf that are NOT in the MNE data structure... let's identify those as well. 
    _, _, unmatched_seeg = match_elec_names(mne_data.ch_names, elec_data.label)

    if method=='wm':
        anode_list, cathode_list, drop_wm_channels, oob_channels = wm_ref(mne_data=mne_data, 
                                                                                       elec_data=elec_data, 
                                                                                       bad_channels=mne_data.info['bads'], 
                                                                                       unmatched_seeg=unmatched_seeg,
                                                                                       site=site)
    elif method=='bipolar':
        pass
    
    # Note that, despite the name, the following function lets you manually set what is being subtracted from what:
    mne_data_reref = mne.set_bipolar_reference(mne_data, 
                          anode=anode_list, 
                          cathode=cathode_list,
                          copy=True)
    mne_data_reref.drop_channels(drop_wm_channels)
    mne_data_reref.drop_channels(oob_channels)

    right_seeg_names = [i for i in mne_data_reref.ch_names if i.startswith('r')]
    left_seeg_names = [i for i in mne_data_reref.ch_names if i.startswith('l')]
    sEEG_mapping_dict = {f'{x}':'seeg' for x in left_seeg_names+right_seeg_names}
    mne_data_reref.set_channel_types(sEEG_mapping_dict)

    return mne_data_reref


def make_epochs(load_path=None, elec_data=None, slope=None, offset=None, behav_name=None, behav_times=None, 
baseline_times=None, baseline_dur=0.5, fixed_baseline=(-1.0, 0),
buf_s=1.0, pre_s=-1.0, post_s=1.5, downsamp_factor=2, IED_args=None):
    """

    TODO: allow for a dict of pre and post times so they can vary across evs 
    
    behav_times: dict with format {'event_name': np.array([times])}
    baseline_times: dict with format {'event_name': np.array([times])}
    IED_args: dict with format {'peak_thresh':5, 'closeness_thresh':0.5, 'width_thresh':0.2}


    Parameters
    ----------
    load_path : str
        path to the re-referenced neural data
    elec_data : pandas df 
        dataframe with all the electrode localization information
    slope : float 
        slope used for syncing behavioral and neural data 
    offset : float 
        offset used for syncing behavioral and neural data 
    behav_name : str
        what event are we epoching to? 
    behav_times : dict 
        format {'event_name': np.array([times])}
    baseline_times : dict 
        format {'event_name': np.array([times])}
    method : str 
        how should we reference the data ['wm', 'bipolar']
    site : str 
        where was this data collected? Options: ['MSSM', 'UI', 'Davis']
    baseline_dur : float
        only to be used if baseline_times is not None 
    fixed_baseline : tuple 
        time to use for baselining , only to be used if baseline_times is None 
    buf_s : float 
        time to add as buffer in epochs 
    pre_s : float 
        time to add before baseline event if baseline_times is not None 
    post_d : float 
        time to add after baseline event if baseline_times is not None 
    downsamp_factor : float 
        factor by which to downsample the data 
    IED_args: dict 
        format {'peak_thresh':5, 'closeness_thresh':0.5, 'width_thresh':0.2}

    Returns
    -------
    ev_epochs : mne object 
        mne Epoch object with re-referenced data
    """
    # Load the data 
    mne_data_reref = mne.io.read_raw_fif(load_path, preload=True)
    # Reconstruct the anode list 
    anode_list = [x.split('-')[0] for x in mne_data_reref.ch_names]

    # Filter the list 
    elec_df = elec_data[elec_data.label.str.lower().isin(anode_list)]
    elec_df['label'] = mne_data_reref.ch_names

    # all behavioral times of interest 
    beh_ts = [(x*slope + offset) for x in behav_times]

    # Make events 
    evs = beh_ts
    durs = np.zeros_like(beh_ts).tolist()
    descriptions = [behav_name]*len(beh_ts)

    # Make mne annotations based on these descriptions
    annot = mne.Annotations(onset=evs,
                            duration=durs,
                            description=descriptions)
    mne_data_reref.set_annotations(annot)
    events_from_annot, event_dict = mne.events_from_annotations(mne_data_reref)

    if baseline_times==None: 
        # Then baseline according to fixed baseline
        ev_epochs = mne.Epochs(mne_data_reref, 
                    events_from_annot, 
                    event_id=event_dict, 
                    baseline=fixed_baseline, 
                    tmin=pre_s - buf_s, 
                    tmax=post_s + buf_s, 
                    reject=None, 
                    reject_by_annotation=False,
                    preload=True)
    else: 
        ev_epochs = mne.Epochs(mne_data_reref, 
            events_from_annot, 
            event_id=event_dict, 
            baseline=None, 
            tmin=pre_s - buf_s, 
            tmax=post_s + buf_s, 
            reject=None, 
            reject_by_annotation=False,
            preload=True)

        # Make baseline epochs to use for baselining 
        baseline_ts = [(x*slope + offset) for x in baseline_times]
        # Make events 
        evs = baseline_ts
        durs = np.zeros_like(baseline_ts).tolist()
        descriptions = list(baseline_times.keys())*len(baseline_ts)
        # Make mne annotations based on these descriptions
        annot = mne.Annotations(onset=evs,
                                duration=durs,
                                description=descriptions)
        mne_data_reref.set_annotations(annot)
        events_from_annot, event_dict = mne.events_from_annotations(mne_data_reref)
        rm_baseline_epochs = mne.Epochs(mne_data_reref, 
            events_from_annot, 
            event_id=event_dict, 
            baseline=None, 
            tmin=-buf, 
            tmax=baseline_dur+buf, 
            reject=None, 
            preload=True)

        buf_ix = int(buf_s*ev_epochs.info['sfreq'])
        time_baseline = rm_baseline_epochs._data[:, :, buf_ix:-buf_ix]
        # Subtract the mean of the baseline data from our data 
        ev_epochs._data = lfp_preprocess_utils.mean_baseline_time(ev_epochs._data, time_baseline, mode='mean')

    # Filter and downsample the epochs 
    ev_epochs.resample(sfreq=ev_epochs.info['sfreq']/downsamp_factor)

    IED_times_s = lfp_preprocess_utils.detect_IEDs(ev_epochs, 
                                               peak_thresh=IED_args['peak_thresh'], 
                                               closeness_thresh=IED_args['closeness_thresh'], 
                                               width_thresh=IED_args['width_thresh'])

    # Let's make METADATA to assign each event some features, including IEDs. Add behavior on your own

    event_metadata = pd.DataFrame(columns=list(IED_times_s.keys()), index=np.arange(len(evs)))

    for ch in list(IED_times_s.keys()):
        for ev, val in IED_times_s[ch].items():
            if len(val) > 1:    
                event_metadata[ch].loc[ev] = val
            else:
                if ~np.isnan(val): 
                    event_metadata[ch].loc[ev] = val
        
    ev_epochs.metadata = event_metadata
    # event_metadata

    return ev_epochs


