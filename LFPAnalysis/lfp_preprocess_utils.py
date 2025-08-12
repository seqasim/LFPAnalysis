import warnings
import numpy as np
import re
import difflib 
from mne.preprocessing.bads import _find_outliers
from scipy.stats import kurtosis, zscore
import mne
from glob import glob
from LFPAnalysis import nlx_utils, lfp_preprocess_utils, iowa_utils
import pandas as pd
from mne.filter import next_fast_len
from scipy.signal import find_peaks, peak_widths
import Levenshtein as lev
import os
import warnings
from ast import literal_eval


def mean_baseline_time(data, baseline, mode='zscore'): 
    """
    Baselines time-series data (i.e the iEEG signal) using a mean baseline period. This function is meant to mimic the MNE baseline function 
    when the specific baseline period might change across trials, as MNE doesn't allow baseline period to vary. 

    Note: Your probably won't use this much!! 

    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_times)
        The original time-series data.
    baseline : numpy.ndarray, shape (n_channels, n_baseline_times)
        The baseline data.
    mode : str, optional
        The type of baseline correction to apply. Valid options are 'mean', 'ratio', 'logratio', 'percent', 'zscore', 
        and 'zlogratio'. Default is 'zscore'.

    Returns
    -------
    baseline_corrected : numpy.ndarray, shape (n_channels, n_times)
        The baseline-corrected time-series data.
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

def baseline_avg_TFR(data, baseline, mode='zscore'): 
    """
    Baselines time-frequency data (TFR) using a mean baseline period. 

    This function presumes you're using trial-averaged data. The baseline data should have the same number of channels and frequencies as the original data.

    Parameters
    ----------
    data : numpy.ndarray, shape (n_channels, n_freqs, n_times)
        The original time-frequency data.
    baseline : numpy.ndarray, shape (n_channels, n_freqs, n_baseline_times)
        The baseline data.
    mode : str, optional
        The type of baseline correction to apply. Valid options are 'mean', 'ratio', 'logratio', 'percent', 'zscore', 
        and 'zlogratio'. Default is 'zscore'.

    Returns
    -------
    baseline_corrected : numpy.ndarray, shape (n_channels, n_freqs, n_times)
        The baseline-corrected time-frequency data.
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
        baseline_corrected = 10 * np.log10(data / m)
    elif mode == 'percent':
        baseline_corrected = (data - m) / m 
    elif mode == 'zscore':
        baseline_corrected = (data - m) / std 
    elif mode == 'zlogratio':
        baseline_corrected = np.log10(data / m) / std
    
    return baseline_corrected 

def baseline_trialwise_TFR(data=None, baseline_mne=None, mode='zscore', include_epoch_in_baseline=True,
                            ev_axis=0, elec_axis=1, freq_axis=2, time_axis=3): 
    
    """
    This function zscores the task data and the baseline data together. Then, it subtracts the mean of the z-scored 
    baseline from the task data.  

    TODO: Make this flexible in case the number of events (baseline_mne) ! = (data)
    
    Parameters
    ----------
    data : np.ndarray, shape (n_trials, n_channels, n_freqs, n_times)
        The original time-frequency data.
    baseline_mne : mne.epochs.Epochs or np.ndarray, shape (n_trials, n_channels, n_freqs, n_times)
        The baseline data. If `trialwise` is True, this should contain baseline data for each trial.
    mode : str, optional
        The type of baseline correction to apply. Valid options are 'mean', 'ratio', 'logratio', 'percent', 'zscore', and 'zlogratio'. Default is 'zscore'.
    trialwise : bool, optional
        Whether to baseline each trial separately. Default is True.
    baseline_only : bool, optional
        Whether to only use the baseline data for correction. Default is False. But depends on 'trialwise'.
    ev_axis : int, optional
        The axis corresponding to the event dimension. Default is 0.
    elec_axis : int, optional
        The axis corresponding to the electrode dimension. Default is 1.
    freq_axis : int, optional
        The axis corresponding to the frequency dimension. Default is 2.
    time_axis : int, optional
        The axis corresponding to the time dimension. Default is 3.

    Returns
    -------
    baseline_corrected : np.ndarray, shape (n_trials, n_channels, n_freqs, n_times)
        The baseline-corrected time-frequency data.
    """

    # The reason I want baseline_mne to be an mne input was to specify these axes in a foolproof way for when
    # I am doing all the replication later on. But needs to be more flexible in case input is a numpy array instead:
    if type(baseline_mne) in [mne.epochs.Epochs, mne.time_frequency.tfr.EpochsTFR]:
        # TODO what if you have the same number of events and channels? This will screw up
        elec_axis = np.where(np.array(baseline_mne.data.shape)==len(baseline_mne.ch_names))[0][0]
        ev_axis = np.where(np.array(baseline_mne.data.shape)==baseline_mne.events.shape[0])[0][0]
        freq_axis = np.where(np.array(baseline_mne.data.shape)==baseline_mne.freqs.shape[0])[0][0]
        time_axis = np.where(np.array(baseline_mne.data.shape)==baseline_mne.times.shape[0])[0][0]

        baseline_data = baseline_mne.data
    else:
        ev_axis = ev_axis
        elec_axis = elec_axis
        freq_axis = freq_axis
        time_axis = time_axis

        baseline_data = baseline_mne

    n_channels = data.shape[elec_axis]
    n_freqs = data.shape[freq_axis]
    n_data_trials = data.shape[ev_axis]
    n_data_times = data.shape[time_axis]

    n_baseline_trials = baseline_data.shape[ev_axis]
    n_baseline_times = baseline_data.shape[time_axis]

    # Reshape the data to combine across all trials and time points. resulting dimension is (n_datapoints, n_channels, n_freqs)
    # reshaped_baseline = np.reshape(baseline_data, (n_baseline_trials * n_baseline_times, n_channels, n_freqs))
    reshaped_baseline = baseline_data.transpose(elec_axis, freq_axis, time_axis, ev_axis).reshape(n_channels, n_freqs, n_baseline_times * n_baseline_trials)

    reshaped_data = data.transpose(elec_axis, freq_axis, time_axis, ev_axis).reshape(n_channels, n_freqs, n_data_times * n_data_trials)

    # concatenate acording to (n_channels, n_freqs)
    if include_epoch_in_baseline:
        baseline_data_concat = np.concatenate((reshaped_baseline, reshaped_data), axis=-1) 
    else: 
        baseline_data_concat = reshaped_baseline

    # Compute mean and std across all time points for each frequency and electrode
    m_ = np.nanmean(baseline_data_concat, axis=-1)
    # std_ = np.nanstd(baseline_data_concat, axis=-1)
    # 12/30/2024: I think that computing std across events like this leads to very large denominators and thus very small z-scores
    # conceptually, this is fine, but it's harder to interpret and reviewers will kind of ignorantly complain about small z-scores. 
    # So, I will first compute the mean across timepoints and then compute the std across events
    std_ = np.squeeze(np.nanstd(np.nanmean(baseline_data, axis=time_axis, keepdims=True), axis=ev_axis))

    # 2. Expand the array to time and events 
    m = np.expand_dims(np.expand_dims(m_, axis=m_.ndim), axis=0)
    # 3. Copy the data to every time and event
    m = np.repeat(np.repeat(m, data.shape[time_axis], axis=time_axis), data.shape[ev_axis], axis=0)

    # 4. Do the same for std
    std = np.expand_dims(np.expand_dims(std_, axis=std_.ndim), axis=0)
    std = np.repeat(np.repeat(std, data.shape[time_axis], axis=time_axis), data.shape[ev_axis], axis=0)

    if mode == 'mean':
        baseline_corrected = data - m
    elif mode == 'ratio':
        baseline_corrected = data / m
    elif mode == 'logratio':
        baseline_corrected = 10 * np.log10(data / m)
    elif mode == 'percent':
        baseline_corrected = (data - m) / m 
    elif mode == 'zscore':
        zscored_data = (data - m) / std 
        # if n_baseline_trials == n_data_trials: # Let's also subtract the trialwise baseline mean from the zscored data
        #     # ref: (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5309795/)

        #     m = np.expand_dims(np.expand_dims(m_, axis=m_.ndim), axis=0)
        #     # 3. Copy the data to every time and event
        #     m = np.repeat(np.repeat(m, baseline_data.shape[time_axis], axis=time_axis), baseline_data.shape[ev_axis], axis=0)

        #     # 4. Do the same for std
        #     std = np.expand_dims(np.expand_dims(std_, axis=std_.ndim), axis=0)
        #     std = np.repeat(np.repeat(std, baseline_data.shape[time_axis], axis=time_axis), baseline_data.shape[ev_axis], axis=0)

        #     zscored_baseline = (baseline_data-m) / std
            
        #     m_b = np.nanmean(zscored_baseline, axis=(time_axis, ev_axis))
        #     m_b = np.expand_dims(np.expand_dims(m_b, axis=m_b.ndim), axis=0)
        #     m_b = np.repeat(np.repeat(m_b, zscored_data.shape[time_axis], axis=time_axis), zscored_data.shape[ev_axis], axis=0)
            
        #     baseline_corrected = zscored_data - m_b
        # else:
        baseline_corrected = zscored_data
    elif mode == 'zlogratio':
        baseline_corrected = np.log10(data / m) / std

    return baseline_corrected

# def baseline_trialwise_TFR_OLD(data=None, baseline_mne=None, mode='zscore', 
#                             trialwise=True, baseline_only=False, 
#                             ev_axis=0, elec_axis=1, freq_axis=2, time_axis=3): 
    
#     """
#     Meant to mimic the mne baseline
#     for TFR but when the specific baseline period might change across trials. 

#     This presumes you're using trial-level data (check dimensions)
    
#     Parameters
#     ----------
#     data : np.ndarray, shape (n_trials, n_channels, n_freqs, n_times)
#         The original time-frequency data.
#     baseline_mne : mne.epochs.Epochs or np.ndarray, shape (n_trials, n_channels, n_freqs, n_times)
#         The baseline data. If `trialwise` is True, this should contain baseline data for each trial.
#     mode : str, optional
#         The type of baseline correction to apply. Valid options are 'mean', 'ratio', 'logratio', 'percent', 'zscore', and 'zlogratio'. Default is 'zscore'.
#     trialwise : bool, optional
#         Whether to baseline each trial separately. Default is True.
#     baseline_only : bool, optional
#         Whether to only use the baseline data for correction. Default is False. But depends on 'trialwise'.
#     ev_axis : int, optional
#         The axis corresponding to the event dimension. Default is 0.
#     elec_axis : int, optional
#         The axis corresponding to the electrode dimension. Default is 1.
#     freq_axis : int, optional
#         The axis corresponding to the frequency dimension. Default is 2.
#     time_axis : int, optional
#         The axis corresponding to the time dimension. Default is 3.

#     Returns
#     -------
#     baseline_corrected : np.ndarray, shape (n_trials, n_channels, n_freqs, n_times)
#         The baseline-corrected time-frequency data.
#     """


#     if (baseline_only==False) & (data is not None):
#         if type(baseline_mne) in [mne.epochs.Epochs, mne.time_frequency.tfr.EpochsTFR]:
#             baseline_data = np.concatenate((baseline_mne.data, data), axis=-1)
#         else: 
#             # ugly but flexible bc i don't want to break people's analyses that assume mne input
#             baseline_data = np.concatenate((baseline_mne, data), axis=-1)
#     else: 
#         # Beware - this is super vulnerable to contamination by artifacts/outliers: https://www.sciencedirect.com/science/article/abs/pii/S1053811913009919
#         if type(baseline_mne) in [mne.epochs.Epochs, mne.time_frequency.tfr.EpochsTFR]:
#             baseline_data = baseline_mne.data
#         else:
#             baseline_data = baseline_mne

#     # The reason I want baseline_mne to be an mne input was to specify these axes in a foolproof way for when
#     # I am doing all the replication later on. But needs to be more flexible in case input is a numpy array instead:

#     if type(baseline_mne) in [mne.epochs.Epochs, mne.time_frequency.tfr.EpochsTFR]:
#         elec_axis = np.where(np.array(baseline_mne.data.shape)==len(baseline_mne.ch_names))[0][0]
#         ev_axis = np.where(np.array(baseline_mne.data.shape)==baseline_mne.events.shape[0])[0][0]
#         freq_axis = np.where(np.array(baseline_mne.data.shape)==baseline_mne.freqs.shape[0])[0][0]
#         time_axis = np.where(np.array(baseline_mne.data.shape)==baseline_mne.times.shape[0])[0][0]
#     else:
#         ev_axis = ev_axis
#         elec_axis = elec_axis
#         freq_axis = freq_axis
#         time_axis = time_axis

#     if trialwise:
#         if baseline_data.shape[0] != data.shape[0]:
#             return print('Baseline data and data must have the same number of trials')
            
#         # Create an array of the mean and standard deviation of the power values across the session
#         # 1. Compute the mean across time points and across trials 
#         m = np.nanmean(baseline_data, axis=(time_axis, axis=ev_axis))
#         # 1. Compute the std across time points for every trial
#         std = np.nanstd(baseline_data, axis=(time_axis, axis=ev_axis))
#     elif baseline_only:
#         # We can compute the mean and std by concatenating all of our baseline data! 

#         # manually reshape
#         baseline_data_reshaped = np.zeros([baseline_data.shape[1], baseline_data.shape[2], baseline_data.shape[-1]*baseline_data.shape[0]])
        
#         for ev in range(baseline_data.shape[0]):
#             ix1 = baseline_data.shape[-1]*ev 
#             ix2 = ix1 + baseline_data.shape[-1]
#             baseline_data_reshaped[:, :, ix1:ix2] = baseline_data[ev, : ,: ,:]

#         # Now we can compute the mean and std across trials and time points all at once 
#         m = np.nanmean(baseline_data_reshaped, axis=-1)
#         std = np.nanstd(baseline_data_reshaped, axis=-1)
#     else:
#         raise ValueError('If baselining across a session then you dont want to concatenate baseline and data. Set baseline_only=True or trialwise=True')

#     # 2. Expand the array
#     m = np.expand_dims(np.expand_dims(m, axis=m.ndim), axis=0)
#     # 3. Copy the data to every time-point
#     m = np.repeat(np.repeat(m, data.shape[time_axis], axis=time_axis), data.shape[ev_axis], axis=0)

#     # 2. Expand the array
#     std = np.expand_dims(np.expand_dims(std, axis=std.ndim), axis=0)
#     # 3. Copy the data to every time-point
#     std = np.repeat(np.repeat(std, data.shape[time_axis], axis=time_axis), data.shape[ev_axis], axis=0)

#     if mode == 'mean':
#         baseline_corrected = data - m
#     elif mode == 'ratio':
#         baseline_corrected = data / m
#     elif mode == 'logratio':
#         baseline_corrected = np.log10(data / m)
#     elif mode == 'percent':
#         baseline_corrected = (data - m) / m 
#     elif mode == 'zscore':
#         baseline_corrected = (data - m) / std 
#     elif mode == 'zlogratio':
#         baseline_corrected = np.log10(data / m) / std

#     # # Can subtract the mean of the baseline period, only, if using the whole trial to z-score (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5309795/)
#     # if (baseline_only==False):
#     #     # 1. Compute the mean across time points and across trials 
#     #     m = np.nanmean(baseline_data, axis=(time_axis))
        

#     return baseline_corrected 

    

def baseline_TFR_permute(data=None, baseline_mne=None, mode='zscore', num_samples=1000,
                            ev_axis=0, elec_axis=1, freq_axis=2, time_axis=3): 
    
    """
    This function samples from all the baseline periods N times with replacement
    and computes the mean and std for normalization of task-related activity. 
    
    Parameters
    ----------
    data : np.ndarray, shape (n_trials, n_channels, n_freqs, n_times)
        The original time-frequency data.
    baseline_mne : mne.epochs.Epochs or np.ndarray, shape (n_trials, n_channels, n_freqs, n_times)
        The baseline data. If `trialwise` is True, this should contain baseline data for each trial.
    mode : str, optional
        The type of baseline correction to apply. Valid options are 'mean', 'ratio', 'logratio', 'percent', 'zscore', and 'zlogratio'. Default is 'zscore'.
    trialwise : bool, optional
        Whether to baseline each trial separately. Default is True.
    baseline_only : bool, optional
        Whether to only use the baseline data for correction. Default is False. But depends on 'trialwise'.
    ev_axis : int, optional
        The axis corresponding to the event dimension. Default is 0.
    elec_axis : int, optional
        The axis corresponding to the electrode dimension. Default is 1.
    freq_axis : int, optional
        The axis corresponding to the frequency dimension. Default is 2.
    time_axis : int, optional
        The axis corresponding to the time dimension. Default is 3.

    Returns
    -------
    baseline_corrected : np.ndarray, shape (n_trials, n_channels, n_freqs, n_times)
        The baseline-corrected time-frequency data.
    """


    if type(baseline_mne) in [mne.time_frequency.tfr.EpochsTFR]:
        baseline_data = baseline_mne.data
    else:
        baseline_data = baseline_mne

    # The reason I want baseline_mne to be an mne input was to specify these axes in a foolproof way for when
    # I am doing all the replication later on. But needs to be more flexible in case input is a numpy array instead:

    if type(baseline_mne) in [mne.time_frequency.tfr.EpochsTFR]:
        elec_axis = np.where(np.array(baseline_mne.data.shape)==len(baseline_mne.ch_names))[0][0]
        ev_axis = np.where(np.array(baseline_mne.data.shape)==baseline_mne.events.shape[0])[0][0]
        freq_axis = np.where(np.array(baseline_mne.data.shape)==baseline_mne.freqs.shape[0])[0][0]
        time_axis = np.where(np.array(baseline_mne.data.shape)==baseline_mne.times.shape[0])[0][0]
    else:
        ev_axis = ev_axis
        elec_axis = elec_axis
        freq_axis = freq_axis
        time_axis = time_axis

    # Slow loop
    m = np.zeros([baseline_data.shape[elec_axis], baseline_data.shape[freq_axis]])
    std = np.zeros([baseline_data.shape[elec_axis], baseline_data.shape[freq_axis]])
    for electrode in range(baseline_data.shape[elec_axis]):
        electrode_data = np.take(baseline_data, indices=[electrode], axis=elec_axis)
        for frequency in range(baseline_data.shape[freq_axis]):
            frequency_data = np.squeeze(np.take(electrode_data, indices=[frequency], axis=freq_axis)).flatten()
            # np.take(arr, indices=[1], axis=axis_to_index)
            samples = np.random.choice(frequency_data, num_samples)
            m[electrode, frequency] = np.nanmean(samples)
            std[electrode, frequency] = np.nanstd(samples)

    # electrode_data = np.take(baseline_data, indices=np.arange(baseline_data.shape[elec_axis]), axis=elec_axis)
    # frequency_data = np.squeeze(np.take(electrode_data, indices=np.arange(baseline_data.shape[freq_axis]), axis=freq_axis)).flatten()
    # samples = np.random.choice(frequency_data, size=(num_samples, baseline_data.shape[elec_axis], baseline_data.shape[freq_axis]))
    # m = np.nanmean(samples, axis=0)
    # std = np.nanstd(samples, axis=0)

    # 2. Expand the array
    m = np.expand_dims(np.expand_dims(m, axis=m.ndim), axis=0)
    # 3. Copy the data to every time-point
    m = np.repeat(np.repeat(m, data.shape[time_axis], axis=time_axis), data.shape[ev_axis], axis=0)

    # 2. Expand the array
    std = np.expand_dims(np.expand_dims(std, axis=std.ndim), axis=0)
    # 3. Copy the data to every time-point
    std = np.repeat(np.repeat(std, data.shape[time_axis], axis=time_axis), data.shape[ev_axis], axis=0)

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




def wm_ref(mne_data=None, elec_path=None, bad_channels=None, unmatched_seeg=None, site='MSSM', average=False):
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

    elec_data = load_elec(elec_path, site=site)

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
        oob_elec_ix_manual = [] 
        oob_elec_ix_auto = []
        false_negatives = [] 
        # account for different labeling strategies in manual column
        white_matter_labels = ['wm', 'white', 'whitematter', 'white matter']
        gray_matter_labels = ['gm', 'gray', 'graymatter', 'gray matter']
        out_of_brain_labels = ['oob', 'out of brain']
        manual_col = elec_data.keys().str.lower().str.contains('manual')
        if np.any(manual_col):
            manual_key = elec_data.keys()[elec_data.keys().str.lower().str.contains('manual')][0]
            wm_elec_ix_manual += [ind for ind, data in elec_data[manual_key].str.lower().items() if data in white_matter_labels and elec_data['label'].str.lower()[ind] not in bad_channels]
            oob_elec_ix_manual += [ind for ind, data in elec_data[manual_key].str.lower().items() if data in out_of_brain_labels]
            false_negatives += [ind for ind, data in elec_data[manual_key].str.lower().items() if data in gray_matter_labels]
        else:
            raise IndexError('No Manual Column!')
    
        # else: # this means we haven't doublechecked the electrode locations manually but trust the automatic locations
        #     print('Beware - no manual examination for electrode locations, could include wm or out-of-brain electrodes')
        wm_elec_ix_auto += [ind for ind, data in elec_data['gm'].str.lower().items() if data=='white' and pd.isnull(elec_data[manual_key][ind])]
        oob_elec_ix_auto += [ind for ind, data in elec_data['gm'].str.lower().items() if data=='unknown']

        # # Correct for false negatives in the autodetection that are corrected by manual examination
        # wm_elec_ix_auto = [x for x in wm_elec_ix_auto if x not in false_negatives]
        oob_elec_ix_auto = [x for x in oob_elec_ix_auto if x not in false_negatives]

        # consolidate manual and auto detection 
        wm_elec_ix = np.unique(wm_elec_ix_manual + wm_elec_ix_auto)
        oob_elec_ix = np.unique(oob_elec_ix_manual + oob_elec_ix_auto)
        all_ix = elec_data.index.values
        gm_elec_ix = np.array([x for x in all_ix if x not in wm_elec_ix and x not in oob_elec_ix])

        cathode_list = []
        anode_list = []
        drop_wm_channels = []
        # reference is anode - cathode, so here wm is cathode
        # 6/14/23: sped up this code!
        
        # compute the distance between all electrodes and white matter electrodes
        
        elec_locs = elec_data.loc[gm_elec_ix, ['x', 'y', 'z']].values.astype(float)
        wm_locs = elec_data.loc[wm_elec_ix, ['x', 'y', 'z']].values.astype(float)
        dists = np.linalg.norm(elec_locs[:, None, :] - wm_locs[None, :, :], axis=-1)

        # only keep the ones in the same hemisphere
        hemi_mask = np.stack([(elec_data.loc[wm_elec_ix, 'label'].str.lower().str[0]==x).values for x in elec_data.loc[gm_elec_ix, 'label'].str.lower().str[0]])
        dists[hemi_mask==False] = np.nan

        # get the 3 closest wm electrodes for each gm electrode
        closest_wm_elec_ix = np.argpartition(dists, 3, axis=1)[:, :3]
        # closest_wm_elec_dist = np.take_along_axis(dists, closest_wm_elec_ix, axis=1)

        # get the variance of the 3 closest wm electrodes
        wm_data = mne_data.copy().pick_channels(elec_data.loc[wm_elec_ix, 'label'].str.lower().tolist())._data
        
        wm_vars = wm_data.var(axis=1)
        wm_elec_var = []
        for wm_ix in closest_wm_elec_ix:
            wm_elec_var.append(wm_vars[wm_ix])

        # get the index of the lowest variance electrode
        wm_elec_ix_lowest = np.argmin(wm_elec_var, axis=1)

        # get the name of the lowest amplitude electrode
        wm_elec_name = elec_data.loc[wm_elec_ix[closest_wm_elec_ix[np.arange(len(gm_elec_ix)), wm_elec_ix_lowest]], 'label'].str.lower()

        # get the electrode name
        anode_list = elec_data.loc[gm_elec_ix, 'label'].str.lower().tolist()
        cathode_list = wm_elec_name.tolist()

        # # DEPRECATED: This loop is SLOW AF: is there a way to vectorize this for speed?
        # for elec_ix in gm_elec_ix:
        #     # get the electrode location
        #     elec_loc = elec_data.loc[elec_ix, ['x', 'y', 'z']].values.astype(float)
        #     elec_name = elec_data.loc[elec_ix, 'label'].lower()
        #     # compute the distance to all wm electrodes
        #     wm_elec_dist = np.linalg.norm(elec_data.loc[wm_elec_ix, ['x', 'y', 'z']].values.astype(float) - elec_loc, axis=1)
        #     # get the closest wm electrodes in order
        #     wm_elec_ix_closest = wm_elec_ix[np.argsort(wm_elec_dist)]
        #     # only keep the ones in the same hemisphere: 
        #     wm_elec_ix_closest = [x for x in wm_elec_ix_closest if elec_data.loc[x, 'label'].lower()[0]==elec_name[0]]
        #     # only keep the 3 closest: 
        #     wm_elec_ix_closest = wm_elec_ix_closest[0:4]
        #     # get the variance of the 3 closest wm electrodes
        #     wm_data = mne_data.copy().pick_channels(elec_data.loc[wm_elec_ix_closest, 'label'].str.lower().tolist())._data
        #     wm_elec_var = wm_data.var(axis=1)
        #     # get the index of the lowest variance electrode
        #     wm_elec_ix_lowest = wm_elec_ix_closest[np.argmin(wm_elec_var)]
        #     # get the name of the lowest amplitude electrode
        #     wm_elec_name = elec_data.loc[wm_elec_ix_lowest, 'label'].lower()
        #     # get the electrode name
        #     anode_list.append(elec_name)
        #     cathode_list.append(wm_elec_name)
            
        # Also collect the wm electrodes that are not used for referencing and drop them later
        drop_wm_channels = [x for x in elec_data.loc[wm_elec_ix, 'label'].str.lower() if x not in cathode_list]
        oob_channels = elec_data.loc[oob_elec_ix, 'label'].str.lower().tolist()

        # cathode_list = np.hstack(cathode_list)
        # anode_list = np.hstack(anode_list)

        return anode_list, cathode_list, drop_wm_channels, oob_channels

    elif site == 'UI':
        wm_elec_ix = [ind for ind, data in elec_data['DesikanKilliany'].str.lower().items() if 'white' in data and elec_data['Channel'][ind] not in mne_data.info['bads']]
        all_ix = elec_data.index.values
        gm_elec_ix = np.array([x for x in all_ix if x not in wm_elec_ix])
        wm_elec_ix = np.array(wm_elec_ix)

        cathode_list = []
        anode_list = []
        drop_wm_channels = []
        # reference is anode - cathode, so here wm is cathode
        elec_locs = elec_data.loc[gm_elec_ix, ['mni_x', 'mni_y', 'mni_z']].values.astype(float)
        wm_locs = elec_data.loc[wm_elec_ix, ['mni_x', 'mni_y', 'mni_z']].values.astype(float)
        dists = np.linalg.norm(elec_locs[:, None, :] - wm_locs[None, :, :], axis=-1)

        # only keep the ones in the same hemisphere
        hemi_mask = np.stack([(elec_data.loc[wm_elec_ix, 'label'].str.lower().str[0]==x).values for x in elec_data.loc[gm_elec_ix, 'label'].str.lower().str[0]])
        dists[hemi_mask==False] = np.nan

        # get the 3 closest wm electrodes for each gm electrode
        closest_wm_elec_ix = np.argpartition(dists, 3, axis=1)[:, :3]
        # closest_wm_elec_dist = np.take_along_axis(dists, closest_wm_elec_ix, axis=1)

        # get the variance of the 3 closest wm electrodes
        wm_data = mne_data.copy().pick_channels(elec_data.loc[wm_elec_ix, 'label'].str.lower().tolist())._data
        
        wm_vars = wm_data.var(axis=1)
        wm_elec_var = []
        for wm_ix in closest_wm_elec_ix:
            wm_elec_var.append(wm_vars[wm_ix])

        # get the index of the lowest variance electrode
        wm_elec_ix_lowest = np.argmin(wm_elec_var, axis=1)

        # get the name of the lowest amplitude electrode
        wm_elec_name = elec_data.loc[wm_elec_ix[closest_wm_elec_ix[np.arange(len(gm_elec_ix)), wm_elec_ix_lowest]], 'label'].str.lower()
        
        # get the electrode name
        anode_list = elec_data.loc[gm_elec_ix, 'label'].str.lower().tolist()
        cathode_list = wm_elec_name.tolist()

        # NOTE: This loop is SLOW AF: is there a way to vectorize this for speed?
        # for elec_ix in gm_elec_ix:
        #     # get the electrode location
        #     elec_loc = elec_data.loc[elec_ix, ['mni_x', 'mni_y', 'mni_z']].values.astype(float)
        #     elec_name = elec_data.loc[elec_ix, 'label'].lower()
        #     # compute the distance to all wm electrodes
        #     wm_elec_dist = np.linalg.norm(elec_data.loc[wm_elec_ix, ['mni_x', 'mni_y', 'mni_z']].values.astype(float) - elec_loc, axis=1)
        #     # get the 3 closest wm electrodes
        #     wm_elec_ix_closest = wm_elec_ix[np.argsort(wm_elec_dist)[:4]]
        #     # only keep the ones in the same hemisphere: 
        #     wm_elec_ix_closest = [x for x in wm_elec_ix_closest if elec_data.loc[x, 'label'].lower()[0]==elec_data.loc[elec_ix, 'label'].lower()[0]]
        #     # get the variance of the 3 closest wm electrodes
        #     wm_data = mne_data.copy().pick_channels(elec_data.loc[wm_elec_ix_closest, 'label'].str.lower().tolist())._data
        #     wm_elec_var = wm_data.var(axis=1)
        #     # get the index of the lowest variance electrode
        #     wm_elec_ix_lowest = wm_elec_ix_closest[np.argmin(wm_elec_var)]
        #     # get the name of the lowest amplitude electrode
        #     wm_elec_name = elec_data.loc[wm_elec_ix_lowest, 'label'].lower()
        #     # get the electrode name
        #     anode_list.append(elec_name)
        #     cathode_list.append(wm_elec_name)

        # Also collect the wm electrodes that are not used for referencing and drop them later
        drop_wm_channels = [x for x in elec_data.loc[wm_elec_ix, 'label'].str.lower() if x not in cathode_list]

        return anode_list, cathode_list, drop_wm_channels


def laplacian_ref(mne_data, elec_path, bad_channels, unmatched_seeg=None, site=None):
    """
    Return the cathode list and anode list for mne to use for laplacian referencing.

    In this case, the cathode is the average of the surrounding electrodes. If an edge electrode, it's just bipolar. 

    Parameters
    ----------
    mne_data : MNE Raw object
        MNE Raw object containing the EEG data
    elec_path : str
        Path to the electrode localization file
    bad_channels : list 
        List of bad channels 
    unmatched_seeg : list 
        List of channels that were not in the edf file 
    site : str
        Hospital where the recording took place 

    Returns
    -------
    anode_list : list 
        List of channels to subtract from
    cathode_list : list 
        List of channels to subtract
    """

    # TODO: for someone clever. Note that you have to bypass the mne reference script because that specific a single reference for each electrode.

    # elec_data = load_elec(elec_path)
    # elec_data['bundle'] = np.nan
    # elec_data['bundle'] = elec_data.apply(lambda x: ''.join(i for i in x.label if not i.isdigit()), axis=1)


    # # helper function to perform sort for bipolar electrodes:
    # def sort_strings(strings):
    #     # Create a regular expression pattern to extract the number at the end of each string
    #     pattern = re.compile(r'\d+$')

    #     # Sort the strings using a custom key function
    #     sorted_strings = sorted(strings, key=lambda x: int(pattern.search(x).group()), reverse=False)

    #     return sorted_strings

    # cathode_list = [] 
    # anode_list = [] 

    # if site=='MSSM':

    #     for bundle in elec_data.bundle.unique():
    #         if bundle[0] == 'u':
    #             print('this is a microwire, pass')
    #             continue         
    #         # Isolate the electrodes in each bundle 
    #         bundle_df = elec_data[elec_data.bundle==bundle].sort_values(by='z', ignore_index=True)
    #         all_elecs = bundle_df.label.tolist()
    #         # Sort them by number 
    #         all_elecs = sort_strings(all_elecs)
    #         # make sure these are not bad channels 
    #         all_elecs = [x for x in all_elecs if x not in bad_channels]
    #         # Set the cathodes and anodes 
    #         for i, elec in enumerate(all_elecs):
    #             # Set the bipolar conditions
    #             if i == 0:
    #                 cathode_list.append(elec)
    #                 anode_list.append((all_elecs[i+1])
    #             elif i == len(all_elecs) - 1:
    #                 cathode_list.append(elec)
    #                 anode_list.append((all_elecs[i-1])  
    #             # Set the laplace conditions 
    #             # else:
    #             # TODO: add laplace conditions here

    # return anode_list, cathode_list

def bipolar_ref(elec_path, bad_channels, unmatched_seeg=None, site='MSSM'):
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

    elec_data = load_elec(elec_path, site=site)
    elec_data['bundle'] = np.nan
    if site == 'MSSM':
        elec_data['bundle'] = elec_data.apply(lambda x: ''.join(i for i in x.label if not i.isdigit()), axis=1)
    elif site == 'UI':
        if any(elec_data.keys().str.contains('Array')):
            elec_data['bundle'] = (elec_data['Array'] != elec_data['Array'].shift()).cumsum()
        elif any(elec_data.keys().str.contains('ContactLabel')):
            elec_data['bundle'] = (elec_data['ContactLabel'] != elec_data['ContactLabel'].shift()).cumsum()

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
    oob_elec_ix_manual = [] 
    oob_elec_ix_auto = []
    false_negatives = [] 
    # account for different labeling strategies in manual column
    white_matter_labels = ['wm', 'white', 'whitematter', 'white matter']
    gray_matter_labels = ['gm', 'gray', 'graymatter', 'gray matter']
    out_of_brain_labels = ['oob', 'out of brain']
    manual_col = elec_data.keys().str.lower().str.contains('manual')
    if np.any(manual_col):
        manual_key = elec_data.keys()[elec_data.keys().str.lower().str.contains('manual')][0]
        wm_elec_ix_manual += [ind for ind, data in elec_data[manual_key].str.lower().items() if data in white_matter_labels and elec_data['label'].str.lower()[ind] not in bad_channels]
        oob_elec_ix_manual += [ind for ind, data in elec_data[manual_key].str.lower().items() if data in out_of_brain_labels]
        false_negatives += [ind for ind, data in elec_data[manual_key].str.lower().items() if data in gray_matter_labels]
    else:
        warnings.warn('Warning...........No Manual Column!')

    # else: # this means we haven't doublechecked the electrode locations manually but trust the automatic locations
    #     print('Beware - no manual examination for electrode locations, could include wm or out-of-brain electrodes')
    
    if site == 'MSSM':
        oob_elec_ix_auto += [ind for ind, data in elec_data['gm'].str.lower().items() if data=='unknown']
        wm_elec_ix_auto += [ind for ind, data in elec_data['gm'].str.lower().items() if data=='white' and pd.isnull(elec_data[manual_key][ind])]

    # # Correct for false negatives in the autodetection that are corrected by manual examination
    # wm_elec_ix_auto = [x for x in wm_elec_ix_auto if x not in false_negatives]
    oob_elec_ix_auto = [x for x in oob_elec_ix_auto if x not in false_negatives]

    # consolidate manual and auto detection 
    wm_elec_ix = np.unique(wm_elec_ix_manual + wm_elec_ix_auto)
    oob_elec_ix = np.unique(oob_elec_ix_manual + oob_elec_ix_auto)

    wm_channels = elec_data['label'].str.lower()[wm_elec_ix].tolist()
    oob_channels = elec_data['label'].str.lower()[oob_elec_ix].tolist()

    # helper function to perform sort for bipolar electrodes:
    def sort_strings(strings):
        # Create a regular expression pattern to extract the number at the end of each string
        pattern = re.compile(r'\d+$')

        # Sort the strings using a custom key function
        sorted_strings = sorted(strings, key=lambda x: int(pattern.search(x).group()), reverse=False)

        return sorted_strings

    cathode_list = [] 
    anode_list = [] 
    drop_wm_channels = [] 

    if site=='MSSM':

        for bundle in elec_data.bundle.unique():
            if bundle[0] == 'u':
                print('this is a microwire, pass')
                continue         
            # Isolate the electrodes in each bundle 
            bundle_df = elec_data[elec_data.bundle==bundle].sort_values(by='z', ignore_index=True)
            all_elecs = bundle_df.label.str.lower().tolist()
            # Sort them by number 
            all_elecs = sort_strings(all_elecs)
            # make sure these are not bad channels 
            all_elecs = [x for x in all_elecs if x not in bad_channels]
            # Set the cathodes and anodes 
            cath = all_elecs[1:]
            an = all_elecs[:-1]
            for c, a in zip(cath, an):
            # I need to make sure I drop any channels where both electrodes are in the wm
            # POSSIBLE ALTERNATIVE FOR FUTURE USERS: you can determine if the "virtual" electrode is in gray matter or not.
                if (c in wm_channels) and (a in wm_channels):
                    drop_wm_channels.append(c)
                    drop_wm_channels.append(a)
                    continue
                # I need to make sure I drop any channels where either electrode is out of the brain
                elif (c in oob_channels) or (a in oob_channels):
                    continue
                else:
                    cathode_list.append(c)
                    anode_list.append(a)
            # # I need to make sure I drop any channels where both electrodes are in the wm
            # if cath in wm_channels and an in wm_channels:
            #     drop_wm_channels.append(cath)
            #     drop_wm_channels.append(an)
            #     continue
            # # I need to make sure I drop any channels where either electrode is out of the brain
            # elif cath in oob_channels or an in oob_channels:
            #     continue
            # else:
            #     cathode_list = cathode_list + cath
            #     anode_list = anode_list + an

    elif site=='UI':

        for bundle in elec_data.bundle.unique():
            # Isolate the electrodes in each bundle 
            bundle_df = elec_data[elec_data.bundle==bundle].sort_values(by='Contact', ignore_index=True)
            all_elecs = bundle_df.label.tolist()
            # make sure these are not bad channels 
            all_elecs = [x for x in all_elecs if x not in bad_channels]
            # Set the cathodes and anodes 
            cath = all_elecs[1:]
            an = all_elecs[:-1]
            for c, a in zip(cath, an):
            # I need to make sure I drop any channels where both electrodes are in the wm
                if (c in wm_channels) and (a in wm_channels):
                    drop_wm_channels.append(c)
                    drop_wm_channels.append(a)
                    continue
                # I need to make sure I drop any channels where either electrode is out of the brain
                elif (c in oob_channels) or (a in oob_channels):
                    continue
                else:
                    cathode_list.append(c)
                    anode_list.append(a)

    return anode_list, cathode_list, drop_wm_channels, oob_channels


def match_elec_names(mne_names, loc_names, method='levenshtein'):
    """
    The electrode names read out of the edf file do not always match those 
    in the pdf (used for localization). This could be error on the side of the tech who input the labels, 
    or on the side of MNE reading the labels in. Usually there's a mixup between lowercase 'l' and capital 'I', or between 'R' and 'P'... 

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

    # check if the number of electrodes in the csv file matches the number of electrodes in the mne file. 
    # it is ok if there are, but we 
    if len(mne_names) > len(loc_names):
        print('Number of electrodes in the mne file is greater than the number of electrodes in the localization file')
        diff_elec_count = True
    elif len(mne_names) < len(loc_names):
        print('Number of electrodes in the mne file is less than the number of electrodes in the localization file')
        diff_elec_count = True
    else: 
        diff_elec_count = False

    # Check which electrode names are in the loc but not the mne
    unmatched_names = list(set(loc_names) - set(mne_names))

    # # macro electrodes start with 'r' or 'l' - find the macro elecs in the mne names which are not in the localization data
    unmatched_seeg = [x for x in unmatched_names if x[0] in ['r', 'l']]

    matched_elecs = []
    replaced_elec_names = []
    cutoff=0.49
    if method=='levenshtein':
        for elec in unmatched_seeg:
            all_lev_ratios = [(x, lev.ratio(elec, x)) for x in mne_names]
            # 9/8/23: Sometimes this algo fails and returns multiple matches with equal distances from the real name: MANUALLY DO THE TIEBREAKER
            lev_df = pd.DataFrame(sorted(all_lev_ratios, key=lambda x: x[1]), columns=['name', 'lev_score'])
            max_lev = lev_df.loc[lev_df['lev_score']==lev_df['lev_score'].max()]
            if max_lev.shape[0] > 1:
                # change all non-leading l's to 'i's and 
                print(max_lev)       
                match_name = input(f'We have too many possible matches for {elec}! Select one manually from these candidates:')
                match = [x for x in all_lev_ratios if x[0]==match_name][0]
            else: 
                match = sorted(all_lev_ratios, key=lambda x: x[1])[-1] # Get the tuples back sorted by highest lev ratio, and pick the first tuple
                match_name = match[0] # Get the actual matched name back 
            # # Make sure the string length matches 
            # ix = -1
            # while len(elec) != len(match_name):
            #     ix -= 1
            #     match = sorted(all_lev_ratios, key=lambda x: x[1])[ix]
            #     match_name = match[0]
            # Make sure this wasn't incorrectly matched to a similar named channel on the same probe with a different NUMBER
            ix = -1
            while int(list(filter(str.isdigit, elec))[0])  != int(list(filter(str.isdigit, match_name))[0]): 
                ix -= 1
                try:
                    match = sorted(all_lev_ratios, key=lambda x: x[1])[ix]
                except IndexError:
                    if diff_elec_count:
                        print('Could not find a match for this electrode. It is likely that the number of electrodes in the mne file does not match the number of electrodes in the localization file.')
                        break
                    else:
                        return IndexError('Could not find a match for this electrode, and its not because the number of electrodes in the mne file does not match the number of electrodes in the localization file.')
                match_name = match[0]
            # Make sure we aren't replacing a legit channel name: 
            ix = -1
            while match_name in list(loc_names):
                ix -= 1
                try:
                    match = sorted(all_lev_ratios, key=lambda x: x[1])[ix]
                except IndexError:
                    if diff_elec_count:
                        print('Could not find a match for this electrode. It is likely that the number of electrodes in the mne file does not match the number of electrodes in the localization file.')
                        break
                    else:
                        return IndexError('Could not find a match for this electrode, and its not because the number of electrodes in the mne file does not match the number of electrodes in the localization file.')
                match_name = match[0]
            if match[1] < cutoff: 
                print(f"Could not find a match for {elec}.")
            else: 
                # agree on one name: the localization name 
                new_mne_names[mne_names.index(match[0])] = elec
                replaced_elec_names.append(match[0])
                matched_elecs.append(elec)
    else:
        # use string matching logic to try and determine if they are just misspelled (often i's and l's are mixed up)
        # (this is a bit of a hack, but should work for the most part)

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

def detect_misc_artifacts(mne_data, peak_thresh=6):
    """
    This function detects artifacts (sharp transients) in the LFP signal automatically. 
    
    """
    # 1. take the gradient of the signal: 
    gradient_signal = np.gradient(mne_data.copy()._data, axis=-1)

    # 2. zscore the gradient of the signal:
    zscored_gradient = zscore(gradient_signal, axis=-1)

    # 3. find where the zscored gradient is above 5
    artifact_samps = np.where(np.abs(zscored_gradient) >= peak_thresh)

    artifact_samps_dict = {f'{x}':np.nan for x in mne_data.ch_names}
    artifact_sec_dict = {f'{x}':np.nan for x in mne_data.ch_names}

    for ch_ in mne_data.ch_names:
        artifact_samps_dict[ch_] = artifact_samps[1][artifact_samps[0] == mne_data.ch_names.index(ch_)]
        artifact_sec_dict[ch_] = (artifact_samps_dict[ch_] / mne_data.info['sfreq'])

    return artifact_sec_dict

def detect_IEDs(mne_data, peak_thresh=5, closeness_thresh=0.25, width_thresh=0.2): 
    """
    This function detects IEDs in the LFP signal automatically. Alternative to manual marking of each ied. 

    From: https://academic.oup.com/brain/article/142/11/3502/5566384

    Method 1: 
    1. Bandpass filter in the [25-80] Hz band. 
    2. Rectify. 
    3. Find filtered envelope > 3. 
    4. Eliminate events with peaks with unfiltered envelope < 3. 
    5. Eliminate close IEDs (peaks within 250 ms). 
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
    elif type(mne_data) == mne.io.fiff.raw.Raw: 
        # , mne.io.edf.edf.RawEDF - probably should never include EDF data directly here. 
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
            IED_samps, _ = find_peaks(sig, height=peak_thresh, distance=closeness_thresh * sr)

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
                IED_samps, _ = find_peaks(sig[event, :], height=peak_thresh, distance=closeness_thresh * sr)
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


# def detect_IEDs_2(mne_data, bandwidth = np.array([10, 60])): 
#     """

    # NOTE: THIS IS TOTAL OVERKILL. Someone can feel free to overhaul this and use it if the IED detection above does not work well. 

#     Here I am going to try and write a better IED detection function that doesn't miss as many as the last one. 

#     https://link.springer.com/article/10.1007/s10548-014-0379-1#Sec2

#     1. Each channel was zero-phase filtered in 10–60 Hz band using combination of high-pass and low-pass 8th order type II Chebyshev digital filters (with stop-band ripple).
#     2.  Instant envelope of each filtered channel was calculated using absolute value of Hilbert transform
#     3. The algorithm estimates the statistical distribution of the envelope and identifies a threshold value, which enables discrimination of spikes from background activity.
#     4. The signal envelope was analysed using a moving window with a segment width of 5 s and 80 % overlap between consecutive segments. 
#     5. The statistical distribution of the envelope was calculated for each segment and approximated with a log-normal fit
#     6. Therefore, mode and median of the log-normal distribution was used to define a threshold that discriminates segments with spikes from the segments with background activity
#     7. mode = np.exp(mu - std**2), median = np.exp(mu) 
#     8. threshold = 3.65 * [mode + median] (from .m code) 
#     """

#     # What type of data is this? Continuous or epoched? 
#     if type(mne_data) == mne.epochs.Epochs:
#         data_type = 'epoch'
#         n_times = mne_data._data.shape[-1]
#     elif type(mne_data) in [mne.io.fiff.raw.Raw, mne.io.edf.edf.RawEDF]: 
#         data_type = 'continuous'
#         n_times = mne_data._data.shape[1]
#     else: 
#         data_type = 'continuous'
#         n_times = mne_data._data.shape[1]  
        

#     signal = mne_data.copy()._data

#     # bandwidth in Hz
#     sr = mne_data.info['sfreq']

#     # Calculate filter parameters
#     Wp = 2 * bandwidth / sr
#     Ws = 2 * bandwidth / sr + 0.1
#     Rp = 6
#     Rs = 60

#     # Calculate filter order and cutoff frequency
#     n, Ws = signal.cheb2ord(Wp, Ws, Rp, Rs)
#     bl, al = signal.cheby2(n, Rs, Ws)

#     # Apply the filter using filtfilt function
#     filtered_signal = signal.filtfilt(bl, al, signal)

#     n_fft = next_fast_len(n_times)

#     # Hilbert bandpass amplitude 
#     envelope = np.abs(hilbert(envelope=True, n_fft=n_fft, n_jobs=-1))


#     # # Given parameters
#     k1 = 3.65 
#     k2 = k1  
#     k3 = 0  
#     polyspike_union_time = 0.12  
#     ti_switch = 1  

#     # Given parameters
#     noverlap = 4*sr  
#     winsize = 5*sr 
#     signal_length = len(signal)  

#     # Calculate index array based on noverlap value
#     if noverlap < 1:
#         step = int(round(winsize * (1 - noverlap)))
#     else:
#         step = winsize - int(noverlap)
        
#     index = np.arange(0, signal_length - winsize + 1, step)


#     # Estimation of segment's distribution using MLE
#     phat = []
#     for k in range(len(index)):
#         segment = envelope[index[k]:index[k] + winsize]
#         segment = segment[segment > 0]  # Remove non-positive values
#         phat.append([np.mean(np.log(segment)), np.std(np.log(segment))])

#     phat = np.array(phat)

#     # Filtering phat using filtfilt
#     r = len(envelope) / len(index)
#     n_average = winsize / fs

#     if round(n_average * fs / r) > 1:
#         phat = np.array([np.convolve(row, np.ones(int(round(n_average * fs / r))) / int(round(n_average * fs / r)), mode='same') for row in phat])

#     # Interpolation of thresholds value to threshold curve
#     phat_int = np.zeros((len(envelope), 2))
#     if len(phat) > 1:
#         phat_int[:, 0] = interp1d(index + round(winsize / 2), phat[:, 0], kind='slinear', fill_value='extrapolate')(np.arange(index[0] + round(winsize / 2), index[-1] + round(winsize / 2) + 1))
#         phat_int[:, 1] = interp1d(index + round(winsize / 2), phat[:, 1], kind='slinear', fill_value='extrapolate')(np.arange(index[0] + round(winsize / 2), index[-1] + round(winsize / 2) + 1))
#     else:
#         phat_int = np.tile(phat, (len(envelope), 1))

#     lognormal_mode = np.exp(phat_int[:, 0] - phat_int[:, 1] ** 2)
#     lognormal_median = np.exp(phat_int[:, 0])
#     lognormal_mean = np.exp(phat_int[:, 0] + (phat_int[:, 1] ** 2) / 2)

#     prah_int = k1 * (lognormal_mode + lognormal_median) - k3 * (lognormal_mean - lognormal_mode)
#     if not (k2 == k1):
#         prah_int_low = k2 * (lognormal_mode + lognormal_median) - k3 * (lognormal_mean - lognormal_mode)
#     else:
#         prah_int_low = prah_int

#     # CDF and PDF of lognormal distribution
#     envelope_cdf = 0.5 + 0.5 * erf((np.log(envelope) - phat_int[:, 0]) / np.sqrt(2 * phat_int[:, 1] ** 2))
#     envelope_pdf = (np.exp(-0.5 * ((np.log(envelope) - phat_int[:, 0]) / phat_int[:, 1]) ** 2) / (envelope * phat_int[:, 1] * np.sqrt(2 * np.pi)))

#     # # Detection of obvious and ambiguous spike

#     def local_maxima_detection(envelope, prah_int, fs, polyspike_union_time, ti_switch, d_decim):
#         marker1 = np.zeros_like(envelope)
#         marker1[envelope > prah_int] = 1
        
#         point = []
#         point.append(np.where(np.diff(np.concatenate(([0], marker1))) > 0)[0])  # start crossing
#         point.append(np.where(np.diff(np.concatenate((marker1, [0]))) < 0)[0])  # end crossing
        
#         if ti_switch == 2:
#             envelope = np.abs(d_decim)
        
#         marker1 = np.zeros_like(envelope, dtype=bool)
#         for k in range(len(point[0])):
#             seg = envelope[point[0][k]:point[1][k] + 1]
#             if len(seg) > 2:
#                 seg_s = np.diff(seg)
#                 seg_s = np.where(np.diff(np.concatenate(([0], np.sign(seg_s)))) < 0)[0]  # positions of local maxima in the section
#                 marker1[point[0][k] + seg_s] = True
#             elif len(seg) <= 2:
#                 s_max = np.argmax(seg)
#                 marker1[point[0][k] + s_max] = True
        
#         pointer = np.where(marker1)[0]
#         state_previous = False
#         start = 0
#         for k in range(len(pointer)):
#             end = min(len(marker1) - 1, pointer[k] + int(polyspike_union_time * fs))
#             seg = marker1[pointer[k] + 1:end]
#             if state_previous:
#                 if np.any(seg):
#                     state_previous = True
#                 else:
#                     state_previous = False
#                     marker1[start:pointer[k] + 1] = True
#             else:
#                 if np.any(seg):
#                     state_previous = True
#                     start = pointer[k]
        
#         point = []
#         point.append(np.where(np.diff(np.concatenate(([0], marker1))) > 0)[0])  # start
#         point.append(np.where(np.diff(np.concatenate((marker1, [0]))) < 0)[0])  # end
        
#         for k in range(len(point[0])):
#             if point[1][k] - point[0][k] > 1:
#                 local_max = pointer[(pointer >= point[0][k]) & (pointer <= point[1][k])]
#                 marker1[point[0][k]:point[1][k] + 1] = False
#                 local_max_val = envelope[local_max]
#                 local_max_poz = np.where(np.diff(np.sign(np.diff(np.concatenate(([0], local_max_val, [0]))))) > 0)[0]
#                 marker1[local_max[local_max_poz]] = True
        
#         return marker1

#     def detection_union(marker1, envelope, union_samples):
#         union_samples = int(np.ceil(union_samples))
#         if union_samples % 2 == 0:
#             union_samples += 1
#         MASK = np.ones(union_samples)
#         marker1_dilated = convolve(marker1.astype(float), MASK, mode='same') > 0  # dilatation
#         marker1_eroded = ~convolve(~marker1_dilated.astype(float), MASK, mode='same').astype(bool)  # erosion
        
#         marker2 = np.zeros_like(marker1)
#         point = []
#         point.append(np.where(np.diff(np.concatenate(([0], marker1_eroded))) > 0)[0])  # start
#         point.append(np.where(np.diff(np.concatenate((marker1_eroded, [0]))) < 0)[0])  # end
        
#         for i in range(len(point[0])):
#             maxp = np.argmax(envelope[point[0][i]:point[1][i] + 1])
#             marker2[point[0][i] + maxp] = 1
        
#         return marker2

#     markers_high = local_maxima_detection(envelope, prah_int, fs, polyspike_union_time, ti_switch, d_decim)
#     markers_high = detection_union(markers_high, envelope, polyspike_union_time * fs)

#     markers_low = local_maxima_detection(envelope, prah_int_low, fs, polyspike_union_time, ti_switch, d_decim)
#     markers_low = detection_union(markers_low, envelope, polyspike_union_time * fs)

#     return markers_low, markers_high

# Below are code that condense the Jupyter notebooks for pre-processing into individual functions. 

def load_elec(elec_path=None, site='MSSM'):
    """
    Load the electrode data from a CSV or Excel file, correct for small idiosyncracies, and return as a pandas dataframe.

    Parameters
    ----------
    elec_path (str): Path to the electrode data file. The file should be in CSV or Excel format.

    Returns
    ----------
    pandas.DataFrame: A dataframe containing the electrode data. The dataframe has columns for the electrode label, the x, y, and z coordinates in MNI space, and any other metadata associated with the electrodes.
    """

    # Load electrode data (should already be manually localized!)
    if elec_path.split('.')[-1] =='csv':
        elec_data = pd.read_csv(elec_path)
    elif elec_path.split('.')[-1] =='xlsx': 
        if site == 'MSSM':
            elec_data = pd.read_excel(elec_path)
        elif site == 'UI':
            # KN excel files: 
            elec_data_all_sheets = pd.read_excel(elec_path, sheet_name=None)
            # Grab the most recent sheet in the dataframe
            sheets = [x for x in list(elec_data_all_sheets.keys()) if 'notes' not in x.lower()]
            elec_data = elec_data_all_sheets[sheets[-1]]
            elec_data.dropna(subset=['Contact'], inplace=True)
            elec_data.reset_index(drop=True, inplace=True)

    # Strip spaces from column headers if they have them: 
    elec_data.columns = elec_data.columns.str.replace(' ', '')

    # Sometimes there's extra columns with no entries: 
    elec_data = elec_data[elec_data.columns.drop(list(elec_data.filter(regex='Unnamed')))]

    if site == 'MSSM':

        if 'NMMlabel' in elec_data.keys(): 
            # This is an annoying naming convention but also totally my fault lol
            elec_data.rename(columns={'NMMlabel':'label'}, inplace=True)

    elif site == 'UI':


        if 'Channel' in elec_data.keys():
            # This is an annoying naming convention for UIowa data
            elec_data['label'] = [f'lfpx{ch}' for ch in elec_data.Channel.values]
        
        if 'mni_x' not in elec_data.keys(): 
            # Check for weird naming of the mni coordinate columns (UIowa)
            elec_data.rename(columns={elec_data.keys()[elec_data.keys().str.lower().str.contains('mnix')].values[0]: 'mni_x',
            elec_data.keys()[elec_data.keys().str.lower().str.contains('mniy')].values[0]: 'mni_y', 
            elec_data.keys()[elec_data.keys().str.lower().str.contains('mniz')].values[0]: 'mni_z'}, inplace=True)

        if 'Desikan-Killianylabel' in elec_data.keys(): 
            elec_data.rename(columns={'Desikan-Killianylabel':'DesikanKilliany'}, inplace=True)

        # # Get rid of unnecessary empty columns 
        # elec_data = elec_data.dropna(axis=1)

        # Assign regions here that match keys for more detailed YBA atlas that we use for MSSM data.

        elec_data['salman_region'] = np.nan
 
        if any(elec_data.keys().str.contains('Destrieux')):

            destr_key = elec_data.keys()[elec_data.keys().str.contains('Destrieux')].values[0]

            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('hippocampus', na=False)] = 'HPC'
            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('amygdala', na=False)] = 'AMY'
            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('temporal', na=False)] = 'Temporal'
            

            # umbrella label captures some operc/triangul/orbital
            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('front_inf', na=False)] = 'dlPFC'

            # rename those 
            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('opercular', na=False)] = 'vlPFC'
            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('triangul', na=False)] = 'vlPFC'

            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('frontopol', na=False)] = 'vmPFC'

            # captures frontal gyrus and lateral, medial, orbital sulci
            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('orbital', na=False)] = 'OFC'
            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('rectus', na=False)] = 'OFC'
            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('front_middle', na=False)] = 'dmPFC'
            elec_data['salman_region'][elec_data[f'{destr_key}'].str.lower().str.contains('insula_ant', na=False)] = 'AINS'
        if any(elec_data.keys().str.contains('Region')):
            
            # make a manual column to assign white matter 
            elec_data['manual'] = np.nan
            # elec_data['manual'][elec_data['Destrieuxlabel'].str.lower().str.contains('white')] = 'white'
            elec_data['manual'][elec_data['Region'].str.lower().str.contains('wm', na=False)] = 'white'
                        
            # Get manual labels for hippocampus and amygdala 
            elec_data['salman_region'][elec_data['Region'].str.lower().str.contains('hippocampus', na=False)] = 'HPC'
            elec_data['salman_region'][elec_data['Region'].str.lower().str.contains('amygdala', na=False)] = 'AMY'
            elec_data['salman_region'][elec_data['Region'].str.lower().str.contains('temporal', na=False)] = 'Temporal'


            # elec_data['salman_region'][elec_data['Destrieuxlabel'].str.lower().str.contains('cingul-ant')] = 'ACC'
            elec_data['salman_region'][elec_data['Region'].str.lower().str.contains('anterior cingulate', na=False)] = 'ACC'

            elec_data['salman_region'][elec_data['Region'].str.lower().str.contains('insula', na=False)] = 'AINS'

            # unique to Iowa: 
            elec_data['salman_region'][elec_data['Region'].str.lower().str.contains('lingual', na=False)] = 'OCC'
            elec_data['salman_region'][elec_data['Region'].str.lower().str.contains('occipital', na=False)] = 'OCC'
            elec_data['salman_region'][elec_data['Region'].str.lower().str.contains('cuneus', na=False)] = 'OCC'

            elec_data['salman_region'][elec_data['Region'].str.lower().str.contains('parietal', na=False)] = 'Parietal'
       
        if any(elec_data.keys().str.contains('Notes')):
            elec_data['manual'][elec_data['Notes'].str.lower().str.contains('outside', na=False)] = 'oob'
            elec_data['manual'][elec_data['Notes'].str.lower().str.contains('ventricle', na=False)] = 'oob'
            elec_data['manual'][elec_data['Notes'].str.lower().str.contains('lesion', na=False)] = 'oob'
            elec_data['manual'][elec_data['Notes'].str.lower().str.contains('cyst', na=False)] = 'oob'
            elec_data['manual'][elec_data['Notes'].str.lower().str.contains('bad', na=False)] = 'oob'










    return elec_data

def make_mne_scalp(load_path=None, overwrite=True, return_data=False):
    """
    Make a mne object from the scalp data file, and save out the sync. 
    Following this step, you can indicate bad electrodes manually.

    This function requires users to input the file format of the raw data.

    Optionally, users can input the names of special channel types as these might be communicated manually rather than hardcoded into the raw data.

    (On that note, a better idea would be for someone to go back and edit the original data to include informative names...)
    
    Parameters
    ----------
    load_path : str
        path to the neural data
    format : str 
        how was this data collected? options: ['edf', 'nlx]
    overwrite: bool 
        whether to overwrite existing data for this person if it exists 
    return_data: bool 
        whether to actually return the data or just save it in the directory 
    eeg_names : list
        list of channel names that pertain to scalp EEG in case the hardcoded options don't work
    resp_names : list 
        list of channel names that pertain to respiration in case the hardcoded options don't work
    ekg_names : list
        list of channel names that pertain to the EKG in case the hardcoded options don't work
    sync_name : str
        provide the sync name in case the hardcoded options don't work
    sync_type : str
        what type of sync signal was used? options: ['photodiode', 'audio', 'ttl']

    Returns
    -------
    mne_data : mne object 
        mne object
    """

    edf_file = glob(f'{load_path}/*.edf')[0]
    mne_data = mne.io.read_raw_edf(edf_file, preload=True)

    # Regex for detecting 10–20/10–10 electrode names
    pattern = re.compile(
        r'^(?:[FTCPOM][pz\d]?|AF\d?|FC\d?|CP\d?|PO\d?|TP\d?|FT\d?|OZ|Fp\d?)$',
        re.IGNORECASE
    )

    def is_scalp_eeg_channel(name):
        # Remove hyphenated bipolar labels and trim
        base = re.split('[- ]', name)[0]
        return bool(pattern.match(base))

    scalp_channels = [ch for ch in mne_data.ch_names if is_scalp_eeg_channel(ch)]

    # restrict to scalp EEG channels
    if not scalp_channels:
        raise ValueError("No scalp EEG channels found in the data. Please check the channel names or the data format.")
    else:
        mne_data.pick_channels(scalp_channels)

    mne_data.info['line_freq'] = 60
    # Notch out 60 Hz noise and harmonics
    mne_data.notch_filter(freqs=(60, 120, 180, 240))

    return mne_data if return_data else mne_data.save(f'{load_path}/scalp_raw.fif', overwrite=overwrite)


def make_mne(load_path=None, elec_path=None, format='edf', site='MSSM', resample_sr = 500, overwrite=True, return_data=False, 
include_micros=False, eeg_names=None, resp_names=None, ekg_names=None, sync_name=None, sync_type='photodiode', seeg_names=None, drop_names=None,
seeg_only=True, check_bad=False):
    """
    Make a mne object from the data and electrode files, and save out the sync. 
    Following this step, you can indicate bad electrodes manually.

    This function requires users to input the file format of the raw data, and the location the data was recorded for site-specific steps.

    Optionally, users can input the names of special channel types as these might be communicated manually rather than hardcoded into the raw data.

    (On that note, a better idea would be for someone to go back and edit the original data to include informative names...)
    
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
    return_data: bool 
        whether to actually return the data or just save it in the directory 
    include_micros : bool
        whether to include the microwire LFP in the LFP data object or not 
    eeg_names : list
        list of channel names that pertain to scalp EEG in case the hardcoded options don't work
    resp_names : list 
        list of channel names that pertain to respiration in case the hardcoded options don't work
    ekg_names : list
        list of channel names that pertain to the EKG in case the hardcoded options don't work
    sync_name : str
        provide the sync name in case the hardcoded options don't work
    drop_names: str
        provide the drop names in case you know certain channels that should be thrown out asap
    seeg_only: bool  (default=True)
        indicate whether you want non seeg channels included

    Returns
    -------
    mne_data : mne object 
        mne object
    """

    if not sync_name:
        warnings.warn(f'No sync name specified - if using an audiovisual sync signal please check load_path to make sure a valid sync was saved out')

    if site == 'MSSM':
        elec_data = load_elec(elec_path)

        if not eeg_names: # If no input, assume the standard EEG montage at MSSM
            eeg_names = ['fp1', 'f7', 't3', 't5', 'o1', 'f3', 'c3', 'p3', 'fp2', 'f8', 't4', 't6', 'o2', 'f4', 'c4', 'p4', 'fz', 'cz', 'pz']
    
    # 1. load the data:
    if format=='edf':
        # MAKE SURE ALL THE EDF CHANNELS HAVE THE SAME SR! See: https://github.com/mne-tools/mne-python/issues/10635
        # EDF data always comes from MSSM AFAIK. Modify this if that changes.

        # This is a big block of data. Have to load first, then split out the sEEG and photodiode downstream. 
        edf_file = glob(f'{load_path}/*.edf')[0]
        mne_data = mne.io.read_raw_edf(edf_file, preload=True)

        if not sync_name:
            if sync_type == 'photodiode':
                # Search for photodiode names if need be
                iteration = 0
                photodiode_options = ['photodiode', 'research', 'sync', 'dc1', 'analog', 'stim', 'trig', 'dc2']
                while (not sync_name) & (iteration<len(photodiode_options)-1):
                    sync_name = next((s for s in mne_data.ch_names if photodiode_options[iteration] in s.lower()), None)
                    iteration += 1
            elif sync_type == 'audio':
                # If/when we implement audio synchronization
                pass
            elif sync_type == 'ttl':
                pass

        if sync_type == 'photodiode':
            # Save out the photodiode channel separately
            mne_data.save(f'{load_path}/photodiode.fif', picks=sync_name, overwrite=overwrite)
        elif sync_type == 'audio':
            pass 
        elif sync_type == 'ttl': 
            print('TTL  used - no need to split out a separate sync channel. Check the .nev file with the neural data.')


        # The electrode names read out of the edf file do not always match those 
        # in the pdf (used for localization). This could be error on the side of the tech who input the labels, 
        # or on the side of MNE reading the labels in. Usually there's a mixup between lowercase 'l' and capital 'I'.
        
        # Sometimes, there's electrodes on the pdf that are NOT in the MNE data structure... let's identify those as well. 
        new_mne_names, _, _ = match_elec_names(mne_data.ch_names, elec_data.label)
        # Rename the mne data according to the localization data
        new_name_dict = {x:y for (x,y) in zip(mne_data.ch_names, new_mne_names)}
        mne_data.rename_channels(new_name_dict)

        if not seeg_names:
            seeg_names = [i for i in mne_data.ch_names if (((i.startswith('l')) | (i.startswith('r'))) & (i!='research'))]
        sEEG_mapping_dict = {f'{x}':'seeg' for x in seeg_names}

        mne_data.set_channel_types(sEEG_mapping_dict)

        mne_data.info['line_freq'] = 60
        # Notch out 60 Hz noise and harmonics 
        mne_data.notch_filter(freqs=(60, 120, 180, 240))

        # drop EEG and EKG channels
        drop_chans = list(set(mne_data.ch_names)^set(seeg_names))
        mne_data.drop_channels(drop_chans)

        if check_bad == True:
            bads = detect_bad_elecs(mne_data, sEEG_mapping_dict)
            mne_data.info['bads'] = bads

        # Resample
        if resample_sr is not None: 
            mne_data.resample(sfreq=resample_sr, npad='auto', n_jobs=-1)
            
        mne_data.save(f'{load_path}/lfp_data.fif', picks=seeg_names, overwrite=overwrite)

    elif format =='nlx': 
        # This is a pre-split data. Have to specifically load the sEEG and sync individually.
        if site == 'MSSM': 
            # MSSM data seems to sometime have a "_0000.ncs" to "_9999.ncs" appended to the end of the data. 
            pattern = re.compile(r"_\d{4}\.ncs") 
            # This is dumb. It happens for one of two reasons: 
            # 1. Recording is paused. When restarted, NLX generates a new file 
            # 2. Separate recordings are being split due to length (several hours)


            # If it's the latter, something got screwed up - no tasks are that long. 
            # TODO 
            # -----
            # If it's the former, we EITHER need to concatenate multiple, real data files, 
            # DONE 
            # -----
            # OR need to select the file that actually has data in it. 


            # First, let's list the file wihout a number attached. This always comes first: 
            ncs_files = [x for x in glob(f'{load_path}/*.ncs') if not re.search(pattern, x)]
            # Second, let's account for all variants 
            numbered_ncs_files = [x for x in glob(f'{load_path}/*.ncs') if re.search(pattern, x)]
            if not seeg_names:
                seeg_names = [x.split('/')[-1].replace('.ncs','') for x in glob(f'{load_path}/[R,L]*.ncs') if not re.search(pattern, x)]
            try: 
                # Let's see if the original files have the data or the numbered variant: 
                test_load = nlx_utils.load_ncs(ncs_files[0])
            except: 
                print('Data in numbered files')
                # This means that we need to load the the files with the numbers appended
                pattern = re.compile(r"_\d{4}\.ncs")  # regex pattern to match "_0000.ncs" to "_9999.ncs"
                ncs_files = numbered_ncs_files
                seeg_names = [x.split('/')[-1].replace('.ncs','').split('_')[0] for x in glob(f'{load_path}/[R,L]*.ncs') if re.search(pattern, x)]

        elif site == 'UI':
            # here, the filenames are not informative. We have to get subject-specific information from the experimenter
            ncs_files = glob(f'{load_path}/LFP*.ncs')

            # load the connection table, which can come in at least three different forms from Iowa:
            # connect_table_path = glob(f'{elec_path}/*Connection*Table*.csv')
            eeg_names = None
            resp_names = None
            ekg_names = None 
            drop_names = None

            if '_KN' in elec_path: 
                seeg_names = iowa_utils.extract_names_elec_table(elec_path)
                # elec_table_path = glob(f'{elec_path}/*_KN.xlsx')
            elif '_fsparc' in elec_path:
                seeg_table = pd.read_csv(elec_path)
                seeg_names =  [f'LFPx{ch}'.lower() for ch in seeg_table.Channel]
            elif 'Connection' in elec_path:
                eeg_names, resp_names, ekg_names, seeg_names, drop_names = iowa_utils.extract_names_connect_table(connect_table_path[0])

        if not seeg_names: 
            raise NameError('no seeg channels specified')
        else:
            # standardize to lower
            seeg_names = [x.lower() for x in seeg_names]

        # Go through every ncs file and parse the useful data based on the extracted channel names and channel types from above
        signals, srs, ch_name, ch_type = nlx_utils.parse_subject_nlx_data(ncs_files,
        eeg_names=eeg_names, 
        resp_names=resp_names, 
        ekg_names=ekg_names, 
        seeg_names=seeg_names, 
        drop_names=drop_names,
        include_micros=include_micros)

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
                    mne_data_temp.resample(sfreq=target_sr, npad='auto', n_jobs=-1)
                    mne_data_resampled.append(mne_data_temp)
                else: 
                    mne_data = mne_data_temp

            #Because of the resampling, the end timings might not match perfectly:https://github.com/mne-tools/mne-python/issues/8257
            if mne_data_resampled[0].times[-1] > mne_data.times[-1]:
                mne_data_resampled = [x.crop(tmin=0, tmax=mne_data.times[-1]) for x in mne_data_resampled]
            elif mne_data_resampled[0].times[-1] < mne_data.times[-1]:
                mne_data.crop(tmin=0, tmax=mne_data_resampled[0].times[-1])

            mne_data.add_channels(mne_data_resampled)

        # Search for sync names if need be
        if not sync_name: 
            if sync_type == 'photodiode':
                iteration = 0
                photodiode_options = ['photodiode', 'research', 'sync', 'dc1', 'analog', 'stim', 'trig', 'dc2']
                while (not sync_name) & (iteration<len(photodiode_options)-1):
                    sync_name = next((s for s in mne_data.ch_names if photodiode_options[iteration] in s.lower()), None)
                    iteration += 1
            elif sync_type == 'audio': 
                pass 
            elif sync_type == 'ttl': 
                pass

        # if not sync_name:
        #     raise ValueError('Could not find a sync channel')

        mne_data.info['line_freq'] = 60
        # Notch out 60 Hz noise and harmonics 
        mne_data.notch_filter(freqs=(60, 120, 180, 240))

        if sync_type == 'photodiode':
            # Save out the photodiode channel separately
            print(f'Saving photodiode data to {load_path}/photodiode.fif')
            mne_data.save(f'{load_path}/photodiode.fif', picks=sync_name, overwrite=overwrite)
        elif sync_type == 'audio':
            pass
        elif sync_type == 'ttl':
            print('TTL  used - no need to split out a separate sync channel. Check the .nev file with the neural data.')

        new_name_dict = {x:x.replace(" ", "").lower() for x in mne_data.ch_names}
        mne_data.rename_channels(new_name_dict)

        # Save out the respiration channels separately
        if resp_names:
            print(f'Saving respiration data to {load_path}/respiration_data.fif')
            mne_data.save(f'{load_path}/respiration_data.fif', picks=resp_names, overwrite=overwrite)
        
        # Save out the EEG channels separately
        if eeg_names: 
            print(f'Saving EEG data to {load_path}/scalp_eeg_data.fif')
            mne_data.save(f'{load_path}/scalp_eeg_data.fif', picks=eeg_names, overwrite=overwrite)

        # Save out the EEG channels separately
        if ekg_names:
            print(f'Saving EKG data to {load_path}/ekg_data.fif')
            mne_data.save(f'{load_path}/ekg_data.fif', picks=ekg_names, overwrite=overwrite)

        if seeg_only == True:
            drop_chans = list(set([x.lower() for x in mne_data.ch_names])^set(seeg_names))
            mne_data.drop_channels(drop_chans)

        if site == 'MSSM':
            # Sometimes, there's electrodes on the pdf that are NOT in the MNE data structure... let's identify those as well. 
            new_mne_names, _, _ = match_elec_names(mne_data.ch_names, elec_data.label)
            # Rename the mne data according to the localization data
            new_name_dict = {x:y for (x,y) in zip(mne_data.ch_names, new_mne_names)}
            mne_data.rename_channels(new_name_dict)

            seeg_names = new_mne_names

        # if site == 'UI':
        #     # Rename the mne channels to match the connectoin table: 
        #     mapping_name = iowa_utils.rename_mne_channels(mne_data, connect_table_path)
        #     mne_data.rename_channels(mapping_name)
        #     seeg_names = mne_data.ch_names
        
        sEEG_mapping_dict = {f'{x}':'seeg' for x in seeg_names}

        if check_bad == True:
            bads = detect_bad_elecs(mne_data, sEEG_mapping_dict)
            mne_data.info['bads'] = bads

        if resample_sr is not None: 
            mne_data.resample(sfreq=resample_sr, npad='auto', n_jobs=-1)
        print(f'Saving LFP data to {load_path}/lfp_data.fif')
        mne_data.save(f'{load_path}/lfp_data.fif', picks=seeg_names, overwrite=overwrite)

    if return_data==True:
        return mne_data


def ref_mne(mne_data=None, elec_path=None, method='wm', site='MSSM'):
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

    elec_data = load_elec(elec_path, site=site)

    # Sometimes, there's electrodes on the pdf that are NOT in the MNE data structure... let's identify those as well. 
    _, _, unmatched_seeg = match_elec_names(mne_data.ch_names, elec_data.label)
  
    if method=='wm':
        anode_list, cathode_list, drop_wm_channels, oob_channels = wm_ref(mne_data=mne_data, 
                                                                                       elec_path=elec_path, 
                                                                                       bad_channels=mne_data.info['bads'], 
                                                                                       unmatched_seeg=unmatched_seeg,
                                                                                       site=site)
    elif method=='bipolar':
        anode_list, cathode_list, drop_wm_channels, oob_channels = bipolar_ref(elec_path=elec_path, 
                                               bad_channels=mne_data.info['bads'], 
                                               unmatched_seeg=unmatched_seeg,
                                               site=site)
        
    
    # Note that, despite the name, the following function lets you manually set what is being subtracted from what:
    mne_data_reref = mne.set_bipolar_reference(mne_data, 
                          anode=anode_list, 
                          cathode=cathode_list,
                          copy=True)
    
    # drop the unreferenced channels (oob or bad or wm)
    mne_data_reref.drop_channels([x for x in mne_data_reref.ch_names if '-' not in x])

    # # drop the white matter channels
    # mne_data_reref.drop_channels(drop_wm_channels)
    # mne_data_reref.drop_channels(oob_channels)


    right_seeg_names = [i for i in mne_data_reref.ch_names if i.startswith('r')]
    left_seeg_names = [i for i in mne_data_reref.ch_names if i.startswith('l')]
    sEEG_mapping_dict = {f'{x}':'seeg' for x in left_seeg_names+right_seeg_names}
    mne_data_reref.set_channel_types(sEEG_mapping_dict)

    return mne_data_reref

def _bin_channelwise_times_into_behav_evs(channel_dict_seconds, ev_starts, ev_ends):
    """
    feed in a dictionary of format {['channel_name']: [time1,...n]}
    timepoints should be in seconds
    every key corresponds to a channel in your mne object

    returns a dataframe of these timepoints binned relative to your behavioral epoch of interest
    useful for detecting artifacts and IEDs in the signal prior to epoching and carrying over those
    detections to the epoched data
    
    ev_starts and ev_ends should be the start and end of each epoch in seconds 
    """
    allts = {f'{x}': np.nan for x in channel_dict_seconds.keys()}
    for key in channel_dict_seconds.keys():
        timestamps = channel_dict_seconds[key]
        time_bins = [(a,b) for (a,b) in zip(ev_starts, ev_ends)]

        # Initialize a dictionary to store the assigned timestamps for each time bin
        assigned_timestamps = {bin_index: [] for bin_index in range(len(time_bins))}

        # Iterate through each timestamp and assign it to the appropriate time bin
        for timestamp in timestamps:
            for bin_index, (start, end) in enumerate(time_bins):
                if start <= timestamp <= end:
                    assigned_timestamps[bin_index].append(timestamp - start)
                    break
        allts[key] = assigned_timestamps
    # Turn the dictionary into a metadata dataframe 
    event_metadata = pd.DataFrame(columns=list(channel_dict_seconds.keys()), index=np.arange(len(time_bins)))
    for ch in list(channel_dict_seconds.keys()):
        for ev, val in allts[ch].items():
            if len(val) > 1:    
                event_metadata[ch].loc[ev] = val
            else:
                # if ~np.isnan(val): 
                event_metadata[ch].loc[ev] = val
    # Replace all nan with Nones 
    event_metadata.where(pd.notna(event_metadata), None)
    return event_metadata

def make_epochs(load_path=None, slope=None, offset=None, behav_name=None, behav_times=None,
ev_start_s=0, ev_end_s=1.5, buf_s=1, downsamp_factor=None, IED_args=None, baseline=None, detrend=None):

    # elec_path=None,
    """

    TODO: allow for a dict of pre and post times so they can vary across evs 
    
    behav_times: dict with format {'event_name': np.array([times])}
    baseline_times: dict with format {'event_name': np.array([times])}
    IED_args: dict with format {'peak_thresh':5, 'closeness_thresh':0.5, 'width_thresh':0.2}

    elec_data : pandas df 
        dataframe with all the electrode localization information

    Parameters
    ----------
    load_path : str
        path to the re-referenced neural data
    slope : float 
        slope used for syncing behavioral and neural data 
    offset : float 
        offset used for syncing behavioral and neural data 
    behav_name : str
        what event are we epoching to? 
    behav_times : dict 
        format 
    baseline_times : dict 
        format 
    ev_start_s:

    ev_end_s: 

    method : str 
        how should we reference the data ['wm', 'bipolar']
    site : str 
        where was this data collected? Options: ['MSSM', 'UI', 'Davis']

    buf_s : float 
        time to add as buffer in epochs 
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

    IED_sec_dict = lfp_preprocess_utils.detect_IEDs(mne_data_reref, 
                                            peak_thresh=IED_args['peak_thresh'], 
                                            closeness_thresh=IED_args['closeness_thresh'], 
                                            width_thresh=IED_args['width_thresh'])

    artifact_sec_dict = lfp_preprocess_utils.detect_misc_artifacts(mne_data_reref, 
                                            peak_thresh=IED_args['peak_thresh'])                                        

    # all behavioral times of interest 
    beh_ts = [(float(x)*slope + offset) if x != 'None' else np.nan for x in behav_times]

    # any NaN's (e.g. non-responses) should be removed. make sure to remove from the dataframes during later analysis too. 
    beh_ts = [x for x in beh_ts if ~np.isnan(x)]
    
    # Bin these times into the epoched bins
    ev_starts = [x - ev_start_s for x in beh_ts]
    ev_ends = [x + ev_end_s for x in beh_ts]

    IED_df = _bin_channelwise_times_into_behav_evs(IED_sec_dict, ev_starts, ev_ends)
    artifact_df = _bin_channelwise_times_into_behav_evs(artifact_sec_dict, ev_starts, ev_ends)

    # # save these out as csvs in the load path 
    bads_path = os.path.dirname(load_path)
    IED_df.to_csv(f'{bads_path}/{behav_name}_IED_df.csv')
    artifact_df.to_csv(f'{bads_path}/{behav_name}_artifact_df.csv')

    #  it doesn't make sense to nan the raw data before computations 
    # instead, let's just save the indices relative to the epochs and nan them after all is said 

    # if nan_artifacts_pre_epoch:
    #     # NaN out the data corresponding to 100 ms before and after each IED and each artifact: 
    #     for ch_ix, ch_ in enumerate(mne_data_reref.ch_names):  
    #         sig = mne_data_reref.get_data(picks=[ch_])[0, :]  
    #         ieds_ = list(IED_sec_dict[ch_])
    #         artifacts_ = list(artifact_sec_dict[ch_])
    #         all_nan_evs_ = ieds_ + artifacts_
    #         for ev_ in all_nan_evs_: 
    #             # ev_ix = ev_ * mne_data_reref.info['sr']
    #             # remove 100 ms before 
    #             ev_ix_start = np.floor((ev_ - 0.1) * mne_data_reref.info['sfreq']).astype(int)
    #             ev_ix_end = np.ceil((ev_ + 0.1) * mne_data_reref.info['sfreq']).astype(int)
    #             sig[ev_ix_start:ev_ix_end] = np.nan
            
    #         mne_data_reref._data[ch_ix, :] = sig


    # Make behavioral events.
    onset_beh = beh_ts
    duration_beh = np.zeros_like(beh_ts).tolist()
    descriptions_beh = [behav_name]*len(beh_ts)
    ch_names_beh = []*len(beh_ts)

    # Make mne annotations based on these descriptions
    annot = mne.Annotations(onset=onset_beh,
                            duration=duration_beh,
                            description=descriptions_beh)
                
    mne_data_reref.set_annotations(annot)
    events_from_annot, event_dict = mne.events_from_annotations(mne_data_reref)

    ev_epochs = mne.Epochs(mne_data_reref, 
        events_from_annot, 
        event_id=event_dict, 
        baseline=baseline, 
        tmin=ev_start_s - buf_s, 
        tmax=ev_end_s + buf_s, 
        detrend=detrend, 
        reject=None, 
        reject_by_annotation=False,
        preload=True)
        
    # NOTE: I don't demean the data for DC offsets. This is mainly because any undetected large artifact would skew and screw us 
    # before any of the following pre-processing steps, which would be hard to detect later.

    # Filter and downsample the epochs 
    if downsamp_factor is not None:
        ev_epochs.resample(sfreq=ev_epochs.info['sfreq']/downsamp_factor)
        # rm_baseline_epochs.resample(sfreq=ev_epochs.info['sfreq']/downsamp_factor)

    # # 1/19/24: Let's also look for noisy epochs, which can persist even after notch filtering the whole session. 
    # if check_epoch_line_noise == True:
    #     notch_freqs = [120, 180, 240] 
    #     notch_ranges = np.concatenate([np.arange(x-3,x+4) for x in notch_freqs]).flatten().tolist()
    #     noisy_epochs_dict = {f'{x}':np.nan for x in ev_epochs.ch_names}

    #     for ch_ in ev_epochs.ch_names:
    #         sig = ev_epochs.get_data(picks=[ch_])[:,0,:]
    #         noise_evs = []
    #         # compute the power spectrum
    #         freqs, psds = compute_spectrum(sig, ev_epochs.info['sfreq'], method='welch', avg_type='median')

    #         for event in np.arange(sig.shape[0]):
    #             # Find peaks in the power spectrum
    #             peaks, _ = find_peaks(np.log10(psds[event, :]), prominence=3.)  # Adjust threshold as needed
    #             peak_freqs = freqs[peaks]
    #             # do they intersect with noise ranges?
    #             intersection = set(peak_freqs) & set(notch_ranges)
    #             if intersection:
    #                 noise_evs.append(event)
    #         # ev_epochs.metadata.loc[noise_evs, ch_] = 'noise'
    #         noisy_epochs_dict[ch_] = noise_evs
    #     noise_df = pd.DataFrame    
    #     # save out the noisy epochs 
    #     noise_df.to_csv(f'{bads_path}/noise_df.csv')

    return ev_epochs

# def get_bad_epochs_by_chan(epochs):
#     """
#     Some of the time, we will want to simply identify all the bad epochs (IED, 60Hz) on a given channel to exclude from analysis.
#     If for some reason you need to split this by category of bad channel, rewrite.
#     """
     
#     good_epochs = {f'{x}': np.nan for x in epochs.ch_names}
#     bad_epochs = {f'{x}': np.nan for x in epochs.ch_names}

#     for ch_ix, ch_name in enumerate(epochs.ch_names):
#         ch_data = epochs._data[:, ch_ix:ch_ix+1, :]
#         bad_epochs[ch_name] = np.where(epochs.metadata[epochs.ch_names[ch_ix]].notnull())[0]
#         good_epochs[ch_name] = np.delete(np.arange(ch_data.shape[0]), bad_epochs[ch_name])

#     return good_epochs, bad_epochs

# def get_bad_epochs_annot(epochs): 
#     """
#     We might want to extract the annotations for the bad epochs so we can make mne objects out of just them
#     """

#     onset_60Hz = [] 
#     duration_60Hz = [] 
#     descriptions_60Hz = [] 
#     ch_names_60Hz = []

#     onset_IED = [] 
#     duration_IED = [] 
#     descriptions_IED = [] 
#     ch_names_IED = []

#     # Make bad events.
#     for ch_ix, ch_name in enumerate(epochs.ch_names):
#         # find categories of bad events
#         bad_events_60Hz = np.where(epochs.metadata[ch_name]=='noise')[0]
#         nbad_60Hz = len(bad_events_60Hz)
#         if nbad_60Hz > 0:
#             onset_60Hz+=behav_times[bad_events_60Hz].values.tolist()
#             duration_60Hz+=np.zeros_like(bad_events_60Hz).tolist()
#             descriptions_60Hz+=['bad_events_60Hz'] * nbad_60Hz
#             ch_names_60Hz+=[[ch_name] for x in range(nbad_60Hz)]

#         bad_events_IED = np.where(epochs.metadata[ch_name].apply(lambda x: isinstance(x, list)))[0]
#         nbad_IED = len(bad_events_IED)
#         if nbad_IED > 0:
#             onset_IED+=behav_times[bad_events_IED].values.tolist()
#             duration_IED+=np.zeros_like(bad_events_IED).tolist()
#             descriptions_IED+=['bad_events_IED'] * nbad_IED
#             ch_names_IED+=[[ch_name] for x in range(nbad_IED)]

#     # merge all events and remake the epochs: 
#     bad_onsets =  onset_60Hz + onset_IED
#     bad_duration = duration_60Hz + duration_IED
#     bad_descriptions =  descriptions_60Hz + descriptions_IED
#     bad_ch_names = ch_names_60Hz + ch_names_IED

#     # Make mne annotations based on these descriptions
#     revised_annot = mne.Annotations(onset=bad_onsets,
#                             duration=bad_duration,
#                             description=bad_descriptions,
#                             ch_names=bad_ch_names)

#     return revised_annot

def rename_elec_df_reref(reref_labels, elec_path, site='MSSM'):

    """
    Sometimes we want to filter and relabel our electrode dataframe based on the renamed channels from the re-referenced data 
    """
    elec_data = load_elec(elec_path, site=site)
    
    
    anode_list = [x.split('-')[0] for x in reref_labels]
    cathode_list = [x.split('-')[1] for x in reref_labels]
    
    # these should be equal length dataframes 
    anode_df = elec_data[elec_data.label.str.lower().isin(anode_list)]
    anode_df['label'] =  anode_df.label.apply(lambda x: [a for a in reref_labels if str(x).lower() in a.split('-')[0]][0])

    cathode_df = elec_data[elec_data.label.str.lower().isin(cathode_list)]
    cathode_df['label'] =  cathode_df.label.apply(lambda x: [a for a in reref_labels if str(x).lower() in a.split('-')[1]][0])

    # Instead of just inheriting anode traits by default
    # Let's account for the possibility that one of these electrodes (anode or cathode) could be in white matter
    # get the white matter electrodes 
    wm_elec_ix_auto = []
    wm_elec_ix_manual = [] 
    # account for different labeling strategies in manual column
    white_matter_labels = ['wm', 'white', 'whitematter', 'white matter']
    manual_col = anode_df.keys().str.lower().str.contains('manual')
    if np.any(manual_col):
        manual_key = anode_df.keys()[anode_df.keys().str.lower().str.contains('manual')][0]
        if anode_df[manual_key].dropna().shape[0] == 0:
            # there are no white matter channels 
            return anode_df
        else:
            wm_elec_ix_manual += [ind for ind, data in anode_df[manual_key].str.lower().items() if data in white_matter_labels]
    else:
        warnings.warn('Warning...........No Manual Column!')

    if site == 'MSSM':
        wm_elec_ix_auto += [ind for ind, data in anode_df['gm'].str.lower().items() if data=='white' and pd.isnull(anode_df[manual_key][ind])]

    # consolidate manual and auto detection 
    wm_elec_ix = np.unique(wm_elec_ix_manual + wm_elec_ix_auto)

    wm_anodes = anode_df['label'].str.lower()[wm_elec_ix].tolist()

    # these are the anodes not in white matter
    anode_no_wm = anode_df[(~anode_df.label.str.lower().isin(wm_anodes))]
    
    # reduce to cathods for elecs with anodes in white matter 
    cathodes_no_wm = cathode_df[cathode_df.label.str.lower().isin(wm_anodes)]

    elec_df = pd.concat([anode_no_wm, cathodes_no_wm], ignore_index=True)

    return elec_df
#

def compute_and_baseline_tfr(baseline_event, task_events, freqs, n_cycles, load_path, save_path,
                            IED_artifact_thresh=True, uncaptured_z_thresh=True, output='save', tfr_method='morlet'):
    
    """
    This function computes the TFRs for the baseline and task events of interest, and baselines the task events of interest

    Parameters
    ----------
    baseline_event : dict
        Dictionary with the key being the name of the baseline event, and the value being a list of the start and end time of the baseline event
    task_events : dict
        Dictionary with the key being the name of the task event, and the value being a list of the start and end time of the task event    
    tfr_method : str
        The method to compute the TFR. Options: ['morlet', 'multitaper']
    freqs : array
        The frequencies of interest for the TFR
    n_cycles : float
        The number of cycles for the Morlet wavelet
    load_path : str
        The path to the directory where the epochs are stored
    save_path : str
        The path to the directory where the TFRs will be saved
    IED_artifact_thresh : bool
        If True, will remove 100 ms before and after IEDs and artifacts from the TFRs
    uncaptured_z_thresh : bool
        If True, will iteratively remove absurd z-scores from the TFRs
    output : str
        If 'save', will save the TFRs to the save_path
        If 'return', will return the TFRs
        If 'both', will save and return the TFRs
    
    """
    
    
    baseline_name = list(baseline_event.keys())[0]
    
    # load baseline epochs
    baseline_epochs_reref = mne.read_epochs(f'{load_path}/{baseline_name}-epo.fif', preload=True)
    
    # compute TFR
    baseline_power = baseline_epochs_reref.compute_tfr(method=tfr_method,
                                             freqs=freqs,
                                             n_cycles=n_cycles,
                                             picks=baseline_epochs_reref.ch_names,
                                             n_jobs=-1,
                                             output='power')
                                                       
    # baseline_power  = mne.time_frequency.tfr_morlet(baseline_epochs_reref, 
    #                                       freqs=freqs, 
    #                                       n_cycles=n_cycles, 
    #                                       picks=baseline_epochs_reref.ch_names,
    #                                       use_fft=True, 
    #                                       n_jobs=-1, 
    #                                       output='power', 
    #                                       return_itc=False, 
    #                                       average=False)
    

    # Crop the data to the appropriate 
    baseline_power.crop(tmin=baseline_event[baseline_name][0], 
                        tmax=baseline_event[baseline_name][1])
    
    
    if IED_artifact_thresh:
        # NAN out the bad data
        # THE following will now LOAD in dataframes that indicate IED and artifact time points in your data
        IED_df = pd.read_csv(f'{load_path}/{baseline_name}_IED_df.csv') 
        artifact_df = pd.read_csv(f'{load_path}/{baseline_name}_artifact_df.csv') 


        # Now, let's iterate through each channel, and each ied/artifact, and NaN 100 ms before and after these timepoints
        for ch_ix, ch_name in enumerate(baseline_epochs_reref.ch_names): 
            ied_ev_list = IED_df[ch_name].dropna().index.tolist()
            artifact_ev_list = artifact_df[ch_name].dropna().index.tolist() 
            for ev_ in ied_ev_list: 
                for ied_ in literal_eval(IED_df[ch_name].iloc[ev_]):
                    # remove 100 ms before 
                    ev_ix_start = np.max([0, np.floor((ied_- 0.1) * baseline_epochs_reref.info['sfreq'])]).astype(int)
                    # remove 100 ms after 
                    ev_ix_end = np.min([baseline_power.data.shape[-1], np.ceil((ied_ + 0.1) * baseline_epochs_reref.info['sfreq'])]).astype(int)
                    baseline_power.data[ev_, ch_ix, :, ev_ix_start:ev_ix_end] = np.nan
            for ev_ in artifact_ev_list: 
                for artifact_ in literal_eval(artifact_df[ch_name].iloc[ev_]):
                    # remove 100 ms before 
                    ev_ix_start = np.max([0, np.floor((artifact_- 0.1) * baseline_epochs_reref.info['sfreq'])]).astype(int)
                    # remove 100 ms after
                    ev_ix_end = np.min([baseline_power.data.shape[-1], np.ceil((artifact_ + 0.1) * baseline_epochs_reref.info['sfreq'])]).astype(int)
                    baseline_power.data[ev_, ch_ix, :, ev_ix_start:ev_ix_end] = np.nan
    
    # remove epochs from memory
    del baseline_epochs_reref
    
    # Now we will deal with the task events of interest   
    
    # make the output path for the z-scored TFRs
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for event in task_events.keys():
        
        event_epochs_reref = mne.read_epochs(f'{load_path}/{event}-epo.fif', preload=True)

        temp_pow = event_epochs_reref.compute_tfr(method=tfr_method,
                                             freqs=freqs,
                                             n_cycles=n_cycles,
                                             picks=event_epochs_reref.ch_names,
                                             n_jobs=-1,
                                             output='power')
                    
        # temp_pow = mne.time_frequency.tfr_morlet(event_epochs_reref, 
        #                                          freqs=freqs, 
        #                                          n_cycles=n_cycles,
        #                                          picks=event_epochs_reref.ch_names, 
        #                                          use_fft=True, 
        #                                          n_jobs=-1, 
        #                                          output='power', 
        #                                          return_itc=False, 
        #                                          average=False)
    
        temp_pow.crop(tmin=task_events[event][0], tmax=task_events[event][1])
        
        if IED_artifact_thresh:
            # NAN out the bad data
            # THE following will now LOAD in dataframes that indicate IED and artifact time points in your data
            IED_df = pd.read_csv(f'{load_path}/{event}_IED_df.csv') 
            artifact_df = pd.read_csv(f'{load_path}/{event}_artifact_df.csv') 

            # Now, let's iterate through each channel, and each ied/artifact, and NaN 100 ms before and after these timepoints
            for ch_ix, ch_name in enumerate(event_epochs_reref.ch_names): 
                ied_ev_list = IED_df[ch_name].dropna().index.tolist()
                artifact_ev_list = artifact_df[ch_name].dropna().index.tolist() 
                for ev_ in ied_ev_list: 
                    for ied_ in literal_eval(IED_df[ch_name].iloc[ev_]):
                        # remove 100 ms before 
                        ev_ix_start = np.max([0, np.floor((ied_- 0.1) * event_epochs_reref.info['sfreq'])]).astype(int)
                        # remove 100 ms after
                        ev_ix_end = np.min([temp_pow.data.shape[-1], np.ceil((ied_ + 0.1) * event_epochs_reref.info['sfreq'])]).astype(int)
                        temp_pow.data[ev_, ch_ix, :, ev_ix_start:ev_ix_end] = np.nan
                for ev_ in artifact_ev_list: 
                    for artifact_ in literal_eval(artifact_df[ch_name].iloc[ev_]):
                        # remove 100 ms before 
                        ev_ix_start = np.max([0, np.floor((artifact_- 0.1) * event_epochs_reref.info['sfreq'])]).astype(int)
                        # remove 100 ms after
                        ev_ix_end = np.min([temp_pow.data.shape[-1], np.ceil((artifact_ + 0.1) * event_epochs_reref.info['sfreq'])]).astype(int)
                        temp_pow.data[ev_, ch_ix, :, ev_ix_start:ev_ix_end] = np.nan    
    
        # Compute first pass of baseline
        baseline_corrected_power = baseline_trialwise_TFR(data=temp_pow.data, include_epoch_in_baseline=False, 
                                                                       baseline_mne=baseline_power.data, 
                                                                       mode='zscore', ev_axis=0, elec_axis=1, freq_axis=2, time_axis=3)
            
        # Let's iteratively nan out absurd z-scores (10 std above baseline???) that escaped our artifact detection, noise removal, and baselining
        if uncaptured_z_thresh: 
            absurdity_threshold = 10
            max_iter = 10
            large_z_flag=True 

            iteration = 0

            while (large_z_flag==True) & (iteration<max_iter): 
                print(f'baseline z-score iteration # {iteration}')
                # Baseline by all the baseline periods in the session
                baseline_corrected_power = baseline_trialwise_TFR(data=temp_pow.data, include_epoch_in_baseline=False, 
                                      baseline_mne=baseline_power.data, 
                                      mode='zscore', ev_axis=0, elec_axis=1, freq_axis=2, time_axis=3)

                large_z_mask = np.where(np.abs(baseline_corrected_power)>absurdity_threshold)
                if large_z_mask[0].shape[0] == 0:
                    # no more large z
                    large_z_flag = False
                else:
                    # NaN it out in the event of interest prior to re-running the baseline z-score to prevent
                    # contamination of all z's
                    temp_pow.data[large_z_mask] = np.nan

                iteration +=1
        
        zpow = mne.time_frequency.EpochsTFR(event_epochs_reref.info, baseline_corrected_power, 
                                    temp_pow.times, freqs)

        zpow.metadata = event_epochs_reref.metadata
        
        # check if save_path exists, if not, make the directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if output == 'save':
            zpow.save(f'{save_path}/{event}-tfr.h5', overwrite=True)
        elif output == 'return': 
            return zpow 
        elif output == 'both':
            zpow.save(f'{save_path}/{event}-tfr.h5', overwrite=True)
            return zpow 
        