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

def zscore_TFR_average(data, baseline): 
    
    """
    Meant to mimic the mne baseline (specifically just the zscore for now) 
    for TFR but when the specific baseline period might change across trials. 

    This presumes you're using trial-averaged data (check dimensions)
    
    TODO: make this more general to any kind of baselining ('mean', etc. )
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

def zscore_TFR_across_trials(data, baseline): 
    
    """
    Meant to mimic the mne baseline (specifically just the zscore for now) 
    for TFR but when the specific baseline period might change across trials. 

    This presumes you're using trial-level data (check dimensions)
    
    TODO: make this more general to any kind of baselining ('mean', etc. )
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

def wm_ref(mne_data, loc_data, bad_channels, unmatched_seeg=None, site=None, average=False):
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

    """

    if site == 'MSSM': 
        # Drop the micros and unmatched seeg from here for now....
        drop_from_locs = []
        for ind, data in loc_data['label'].str.lower().items(): 
            if data in unmatched_seeg:
                drop_from_locs.append(ind)
            elif data[0] == 'u':
                drop_from_locs.append(ind)

        loc_data = loc_data.drop(index=drop_from_locs).reset_index(drop=True)

        # get the white matter electrodes and make sure they are note in the bad channel list
        wm_elec_ix_manual = [] 
        wm_elec_ix_auto = []
        if 'Manual Examination' in loc_data.keys():
            wm_elec_ix_manual = wm_elec_ix_manual + [ind for ind, data in loc_data['Manual Examination'].str.lower().items() if data=='wm' and loc_data['label'].str.lower()[ind] not in bad_channels]
            oob_elec_ix = [ind for ind, data in loc_data['Manual Examination'].str.lower().items() if data=='oob']
        else: # this means we haven't doublechecked the electrode locations manually but trust the automatic locations
            wm_elec_ix_auto = wm_elec_ix_auto + [ind for ind, data in loc_data['gm'].str.lower().items() if data=='white' and loc_data['label'].str.lower()[ind] not in bad_channels]
            oob_elec_ix = [ind for ind, data in loc_data['gm'].str.lower().items() if data=='unknown']

        wm_elec_ix = np.unique(wm_elec_ix_manual + wm_elec_ix_auto)
        all_ix = loc_data.index.values
        gm_elec_ix = np.array([x for x in all_ix if x not in wm_elec_ix and x not in oob_elec_ix])

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
            # get the variance of the 3 closest wm electrodes
            wm_data = mne_data.copy().pick_channels(loc_data.loc[wm_elec_ix_closest, 'label'].str.lower().tolist())._data
            wm_elec_var = wm_data.var(axis=1)
            # get the index of the lowest variance electrode
            wm_elec_ix_lowest = wm_elec_ix_closest[np.argmin(wm_elec_var)]
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

    elif site == 'UI':
        wm_elec_ix = [ind for ind, data in loc_data['region_1'].str.lower().items() if 'white' in data and loc_data['Channel'][ind] not in mne_data.info['bads']]
        all_ix = loc_data.index.values
        gm_elec_ix = np.array([x for x in all_ix if x not in wm_elec_ix])
        wm_elec_ix = np.array(wm_elec_ix)

        cathode_list = []
        anode_list = []
        drop_wm_channels = []
        # reference is anode - cathode, so here wm is cathode

        # NOTE: This loop is SLOW AF: is there a way to vectorize this for speed?
        for elec_ix in gm_elec_ix:
            # get the electrode location
            elec_loc = loc_data.loc[elec_ix, ['mni_x', 'mni_y', 'mni_z']].values.astype(float)
            elec_name = loc_data.loc[elec_ix, 'Channel'].lower()
            # compute the distance to all wm electrodes
            wm_elec_dist = np.linalg.norm(loc_data.loc[wm_elec_ix, ['mni_x', 'mni_y', 'mni_z']].values - elec_loc, axis=1)
            # get the 3 closest wm electrodes
            wm_elec_ix_closest = wm_elec_ix[np.argsort(wm_elec_dist)[:4]]
            # only keep the ones in the same hemisphere: 
            wm_elec_ix_closest = [x for x in wm_elec_ix_closest if loc_data.loc[x, 'label'].lower()[0]==loc_data.loc[elec_ix, 'label'].lower()[0]]
            # get the variance of the 3 closest wm electrodes
            wm_data = mne_data.copy().pick_channels(loc_data.loc[wm_elec_ix_closest, 'Channel'].str.lower().tolist())._data
            wm_elec_var = wm_data.var(axis=1)
            # get the index of the lowest variance electrode
            wm_elec_ix_lowest = wm_elec_ix_closest[np.argmin(wm_elec_var)]
            # get the name of the lowest amplitude electrode
            wm_elec_name = loc_data.loc[wm_elec_ix_lowest, 'Channel'].lower()
            # get the electrode name
            anode_list.append(elec_name)
            cathode_list.append(wm_elec_name)

        # Also collect the wm electrodes that are not used for referencing and drop them later
        drop_wm_channels = [x for x in loc_data.loc[wm_elec_ix, 'Channel'].str.lower() if x not in cathode_list]

        return anode_list, cathode_list, drop_wm_channels


def laplacian_ref(mne_data, loc_data, bad_channels, unmatched_seeg=None, site=None):
    """
    Return the cathode list and anode list for mne to use for laplacian referencing.

    In this case, the cathode is the average of the surrounding electrodes. If an edge electrode, it's just bipolar. 

    From here: https://doi.org/10.1016/j.neuroimage.2018.08.020

    """

    pass

def bipolar_ref(loc_data, bad_channels, unmatched_seeg=None, site=None):
    """
    Return the cathode list and anode list for mne to use for bipolar referencing.

    TODO: figure out a renaming convention across sites so that this can be generalized.
    """

    # helper function to perform sort for bipolar electrodes:
    def num_sort(string):
        return list(map(int, re.findall(r'\d+', string)))[0]

    cathode_list = [] 
    anode_list = [] 

    if site=='MSSM':

        for bundle in loc_data.bundle.unique():
            if bundle[0] == 'u':
                print('this is a microwire, pass')
                continue         
            # Isolate the electrodes in each bundle 
            bundle_df = loc_data[loc_data.bundle==bundle].sort_values(by='z', ignore_index=True)
            all_elecs = loc_data.label.tolist()
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

        for bundle in loc_data.bundle.unique():
            # Isolate the electrodes in each bundle 
            bundle_df = loc_data[loc_data.bundle==bundle].sort_values(by='contact', ignore_index=True)
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

def detect_bad_elecs(mne_data, sEEG_mapping_dict): 
    """
    Find outlier channels using a combination of kurtosis, variance, and standard deviation. Also use the loc_data to find channels out of the brain
    
    https://www-sciencedirect-com.eresources.mssm.edu/science/article/pii/S016502701930278X
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7472198/
    https://www.biorxiv.org/content/10.1101/2021.05.14.444176v2.full.pdf

    
    Plot these channels for manual verification. 
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

    # 

    return np.unique(kurt_chans.tolist() + var_chans.tolist() + std_chans.tolist()).tolist()

def detect_IEDs(mne_data, peak_thresh=3, closeness_thresh=0.5, width_thresh=0.2): 
    """
    Great images of interictal and ictal spiking: 
    https://pubmed.ncbi.nlm.nih.gov/32007920/
    
    https://www.sciencedirect.com/science/article/pii/S1059131119304200

    More on IEDs: https://www.frontiersin.org/articles/10.3389/fnhum.2020.00044/full


    This function detects IEDs in the LFP signal automatically. Alternative to manual marking of each ied. 

    Method 1: 
    1. Bandpass filter in the [25-80] Hz band. 
    2. Rectify. 
    3. Find filtered envelope > 3. 
    4. Eliminate events with peaks with unfiltered envelope < 3. 
    5. Eliminate close IEDs (peaks within 500 ms). 
    6. TODO: Compute rise-decay asymmetry (time-to-peak / time between troughs). Must be < 0.5
    (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6821283/)

    """

    sr = mne_data.info['sfreq']
    min_width = width_thresh * sr

    # filter data in beta bands 
    filtered_data = mne_data.copy().filter(25, 80, n_jobs=-1)

    n_times = mne_data._data.shape[1]
    n_fft = next_fast_len(n_times)

    # Hilbert bandpass amplitude 
    filtered_data = filtered_data.apply_hilbert(envelope=True, n_fft=n_fft, n_jobs=-1)

    # Zscore
    filtered_data.apply_function(lambda x: zscore(x, axis=-1))

    ## Rectify
    # filtered_data.apply_function(lambda x: x[x<0] = 0)

    IED_times_s = {f'{x}':np.nan for x in mne_data.ch_names}

    for ch_ in filtered_data.ch_names:
        sig = filtered_data.get_data(picks=[ch_])[0, :]
        
        # Rectify
        sig[sig<0] = 0

        # Find peaks 
        IED_index, _ = find_peaks(sig, height=peak_thresh, prominence=2, distance=closeness_thresh * sr)

        widths = peak_widths(sig, IED_index, rel_height=0.75)

        # print(widths[0])
        too_wide_IEDs = np.where(widths[0] > min_width)[0]

        # print(too_wide_IEDs)
        # IED_z_amp = IED_z_amp['peak_heights']

        # Which peaks are below 3 in z-scored unfiltered signal? 
        small_IEDs = np.where(zscore(mne_data.get_data(picks=[ch_]), axis=-1)[0, IED_index] < 3)[0]

        # Which IEDs follow an initial IED too closely? 
        # follow_IEDs = np.where(((1000* (np.diff(IED_index) / sr)) <= closeness_thresh))[0]+1
        
        elim_IEDs = np.unique(np.hstack([small_IEDs, too_wide_IEDs]))

        IED_index = np.delete(IED_index, elim_IEDs)
        # IED_z_amp = np.delete(IED_z_amp, elim_IEDs)
        IED_s = (IED_index / sr)

        IED_times_s[ch_] = IED_s

    return IED_times_s

    
def make_mne(data_path=None, elec_path=None, save_path=None, format='edf'):
    """
    Make a mne object from the data and electrode files, and save out the photodiode. 
    Following this step, you can indicate bad electrodes manually.
    """

    # 1) load the data:
    if format=='edf':
        edf_file = glob(f'{data_path}/*.edf')[0]
        mne_data = mne.io.read_raw_edf(edf_file, preload=True)
    elif format =='nlx': 
        ncs_files = glob(f'{data_path}/*.ncs')
        for chan_path in ncs_files:
            # This is leftover from Iowa - let's see how channels are named in MSSM data
            chan_name = chan_path.split('/')[-1][:-4]
            ch_num = int(chan_name[4:])
            try:
                fdata = nlx_utils.load_ncs(chan_path)
            except IndexError: 
                print(f'No data in channel {chan_name}')
                continue
            lfp.append(fdata['data'])
            sr.append(fdata['sampling_rate'])
            ch_name.append(str(ch_num))
            unix_time = fdata['time']
        info = mne.create_info(ch_name, np.unique(sr), ch_type)
        mne_data = mne.io.RawArray(lfp, info)
    
    # 2) load the electrode file:
    loc_data = pd.read_csv(elec_path)

    # Sometimes there's extra columns with no entries: 
    loc_data = loc_data[loc_data.columns.drop(list(loc_data.filter(regex='Unnamed')))]

    if format=='edf':
        # The electrode names read out of the edf file do not always match those 
        # in the pdf (used for localization). This could be error on the side of the tech who input the labels, 
        # or on the side of MNE reading the labels in. Usually there's a mixup between lowercase 'l' and capital 'I'.
        
        # Sometimes, there's electrodes on the pdf that are NOT in the MNE data structure... let's identify those as well. 
        new_mne_names, _, _ = match_elec_names(mne_data.ch_names, loc_data.label)
        # Rename the mne data according to the localization data
        new_name_dict = {x:y for (x,y) in zip(mne_data.ch_names, new_mne_names)}
        mne_data.rename_channels(new_name_dict)

    right_seeg_names = [i for i in mne_data.ch_names if i.startswith('r')]
    left_seeg_names = [i for i in mne_data.ch_names if i.startswith('l')]
    sEEG_mapping_dict = {f'{x}':'seeg' for x in left_seeg_names+right_seeg_names}

    mne_data.set_channel_types(sEEG_mapping_dict)


    # 3) Identify line noise
    mne_data.info['line_freq'] = 60

    # Notch out 60 Hz noise and harmonics 
    mne_data.notch_filter(freqs=(60, 120, 180, 240))

    # 4) Save out the photodiode channel separately
    mne_data.save(f'{save_path}/photodiode.fif', picks='dc1', overwrite=True)

    # 5) Clean up the MNE data 

    bads = detect_bad_elecs(mne_data, sEEG_mapping_dict)

    mne_data.info['bads'] = bads

    return mne_data


def ref_mne(mne_data, loc_data, method='wm', site='MSSM'):
    """
    Following this step, you can indicate IEDs manually.
    """
    
    # Check which electrode names are in the loc but not the mne
    unmatched_names = list(set(loc_data.label) - set(mne_data.ch_names))
    # seeg electrodes start with 'r' or 'l' - find the elecs in the mne names which are not in the localization data
    unmatched_seeg = [x for x in unmatched_names if x[0] in ['r', 'l']]

    if method=='wm':
        anode_list, cathode_list, drop_wm_channels, oob_channels = wm_ref(mne_data=mne_data, 
                                                                                       loc_data=loc_data, 
                                                                                       bad_channels=mne_data.info['bads'], 
                                                                                       unmatched_seeg=unmatched_seeg,
                                                                                       site=site)
    elif method=='bipolar':
        pass
    
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

