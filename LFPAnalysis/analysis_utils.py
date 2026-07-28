import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
import math
from functools import lru_cache
from pathlib import Path
import pandas as pd
import mne
import os
import warnings

from .config import WORKING_DTYPE
from .validation import ensure_dependency

# NumPy dtype object resolved once at import (config stores the name string only).
_WORKING_DTYPE = np.dtype(WORKING_DTYPE).type


def _yba_roi_excel_path() -> Path:
    """Return path to the packaged YBA ROI lookup table."""

    return Path(__file__).resolve().parent / "YBA_ROI_labelled.xlsx"


# There are some things that MNE is not that good at, or simply does not do. Let's write our own code for these. 
@lru_cache(maxsize=1)
def _load_yba_roi_labels() -> pd.DataFrame:
    """Load and normalize the packaged YBA ROI lookup table once."""

    file_path = _yba_roi_excel_path()
    yba_roi_labels = pd.read_excel(file_path)
    yba_roi_labels['Long.name'] = (
        yba_roi_labels['Long.name'].astype(str).str.lower().str.replace(" ", "", regex=False)
    )
    return yba_roi_labels


def _normalize_roi_label(value):
    """Normalize electrode label text for atlas lookups."""

    if pd.isna(value):
        return np.nan
    return str(value).lower().replace(" ", "").strip()


def select_rois_picks(elec_data: pd.DataFrame, chan_name: str, manual_col: str = 'collapsed_manual'):
    """Map one channel to a collapsed ROI using lab atlas columns.

    Required / used columns (site-dependent): ``label``, ``NMM``, ``BN246``,
    ``YBA_1``, and optionally ``collapsed_manual``. Lab-specific alias rules —
    not a general atlas.

    Parameters
    ----------
    elec_data : pd.DataFrame
        Electrode data DataFrame.
    chan_name : str
        Channel name.
    manual_col : str, optional
        Manual column name. Default is 'collapsed_manual'.

    Returns
    -------
    str
        ROI label.
    """

    yba_roi_labels = _load_yba_roi_labels()

    chan_rows = elec_data.loc[elec_data.label == chan_name]
    if chan_rows.empty:
        raise KeyError(f"Channel {chan_name!r} was not found in electrode metadata.")

    chan_row = chan_rows.iloc[0]

    roi = np.nan
    nmm_label = _normalize_roi_label(chan_row.get('NMM'))
    bn246_label = _normalize_roi_label(chan_row.get('BN246'))

    # Account for individual differences in labelling: 
    yba_label = _normalize_roi_label(chan_row.get('YBA_1'))
    manual_label = _normalize_roi_label(chan_row.get(manual_col))

    # Only NMM assigns entorhinal cortex 
    if isinstance(nmm_label, str) and 'entorhinal' in nmm_label:
        roi = 'EC'

    # First priority: Use YBA labels if there is no manual label
    if pd.isna(manual_label):
        try:
            roi = yba_roi_labels.loc[yba_roi_labels['Long.name'] == yba_label, 'Custom'].iat[0]
        except IndexError:
            # This is probably white matter or out of brain, but not manually labelled as such
            roi = np.nan
    else:
        # Now look at the manual labels: 
        if isinstance(yba_label, str) and 'unknown' in yba_label:
            # prioritize thalamus labels! Which are not present in YBA for some reason
            if 'thalamus' in manual_label:
                roi = 'THAL'
            else:
                try:
                    roi = yba_roi_labels.loc[yba_roi_labels['Long.name'] == manual_label, 'Custom'].iat[0]
                except IndexError: 
                    # This is probably white matter or out of brain, and manually labelled as such
                    roi = np.nan

    # Next  use BN246 labels if still unlabeled
    if pd.isna(roi):
        # Just use the dumb BN246 label from LeGui, stripping out the hemisphere which we don't care too much about at the moment
        if isinstance(bn246_label, str) and 'hipp' in bn246_label:
            roi = 'HPC'
        elif isinstance(bn246_label, str) and 'amyg' in bn246_label:
            roi = 'AMY'
        elif isinstance(bn246_label, str) and 'ins' in bn246_label:
            roi = 'INS'
        elif isinstance(bn246_label, str) and 'ifg' in bn246_label:
            roi = 'IFG'
        elif isinstance(bn246_label, str) and 'org' in bn246_label:
            roi = 'OFC' 
        elif isinstance(bn246_label, str) and 'mfg' in bn246_label:
            roi = 'dlPFC'
        elif isinstance(bn246_label, str) and 'sfg' in bn246_label:
            roi = 'dmPFC'

    if pd.isna(roi):
        # Just use the dumb NMM label from LeGui, stripping out the hemisphere which we don't care too much about at the moment
        if isinstance(nmm_label, str) and 'hippocampus' in nmm_label:
            roi = 'HPC'
        if isinstance(nmm_label, str) and 'amygdala' in nmm_label:
            roi = 'AMY'
        if isinstance(nmm_label, str) and 'acgc' in nmm_label:
            roi = 'ACC'
        if isinstance(nmm_label, str) and 'mcgc' in nmm_label:
            roi = 'MCC'
        if isinstance(nmm_label, str) and 'ofc' in nmm_label:
            roi = 'OFC'
        if isinstance(nmm_label, str) and 'mfg' in nmm_label:
            roi = 'dlPFC'
        if isinstance(nmm_label, str) and 'sfg' in nmm_label:
            roi = 'dmPFC'  

    if pd.isna(roi):
        # This is mostly temporal gyrus
        roi = 'Unknown'

    return roi

def select_picks_rois(elec_data: pd.DataFrame, roi=None):
    """Select electrodes for specific ROI.
    
    Parameters
    ----------
    elec_data : pd.DataFrame
        Electrode data DataFrame.
    roi : str or list, optional
        ROI name or list of ROI names.
    
    Returns
    -------
    list
        List of electrode labels.
    """

    # Site specific processing: 
    if roi == 'anterior_cingulate':
        # here is my approximation of anterior cingulate in the YBA atlas
        # TODO improve this
        roi = ['cingulate gyrus a', 'cingulate gyrus b', 'cingulate gyrus c']

    if roi == 'entorhinal': 
        # entorhinal is not in the YBA atlas
        picks = elec_data[elec_data.NMM.str.lower().str.contains(roi)].label.tolist()
        return picks

    if isinstance(roi, str):
        picks = elec_data[elec_data.YBA_1.str.lower().str.contains(roi)].label.tolist()
    elif isinstance(roi, list):
        # then assume the user wants to group several regions
        picks_ec = None
        if 'anterior_cingulate' in roi: 
            roi.remove('anterior_cingulate')
            roi += ['cingulate gyrus a', 'cingulate gyrus b', 'cingulate gyrus c']
        elif 'entorhinal' in roi: 
            roi.remove('entorhinal')
            picks_ec =  elec_data[elec_data.NMM.str.lower().str.contains('entorhinal')].label.tolist()
        picks = elec_data[elec_data.YBA_1.str.lower().str.contains('|'.join(roi))].label.tolist()
        if picks_ec is not None: 
            picks += picks_ec

    else:
        # Just grab everything 
        picks = elec_data.label.tolist()
    
    return picks 

def lfp_sta(ev_times: np.ndarray, signal: np.ndarray, sr: float, pre: float, post: float):
    """Compute spike-triggered average for LFP signal.
    
    Parameters
    ----------
    ev_times : np.ndarray
        Event times in seconds.
    signal : np.ndarray
        Signal for averaging.
    sr : float
        Sampling rate.
    pre : float
        Pre-event window in seconds.
    post : float
        Post-event window in seconds.
    
    Returns
    -------
    tuple
        Tuple containing (sta, ste).
    """

    num_evs = len(ev_times)
    ev_in_samples = (ev_times * sr).astype(int)
    pre_in_samples = int(pre  * sr)
    post_in_samples = int(post * sr)
    
    lfp_pre_avg = np.zeros([num_evs, (pre_in_samples + post_in_samples)])
    for sidx in range(0, num_evs):
        idx1 = math.ceil(ev_in_samples[sidx]) - pre_in_samples
        idx2 = math.floor(ev_in_samples[sidx]) + post_in_samples
        if len(range(idx1, idx2)) != (pre_in_samples + post_in_samples):
            continue
        else:
            try:
                lfp_pre_avg[sidx, :] = signal[idx1:idx2]  # - nanmean(raw_lfp(idx1:idx2)); % subtract the mean of the signal
            except ValueError: 
                continue

    sta = np.nanmean(lfp_pre_avg, 0)
    ste = np.nanstd(lfp_pre_avg, 0) / np.sqrt(len(sta))
    return sta, ste


def plot_TFR(data: np.ndarray, freqs: np.ndarray, pre_win: float, post_win: float, sr: float, title: str):
    """Plot time-frequency representation.
    
    Parameters
    ----------
    data : np.ndarray
        TFR data array.
    freqs : np.ndarray
        Frequency array.
    pre_win : float
        Pre-window in seconds.
    post_win : float
        Post-window in seconds.
    sr : float
        Sampling rate.
    title : str
        Plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """

    f, tfr = plt.subplots(1, 1, figsize=[7, 4], dpi=300)

    tfr.imshow(data, aspect='auto', interpolation='bicubic', cmap='RdBu_r', vmin=-3, vmax=3)
    tfr.invert_yaxis()

    tfr.set_yticks(np.arange(0, len(freqs), 4))
    tfr.set_yticklabels(np.round(freqs[np.arange(0, len(freqs), 4)]), fontsize=10)
    tfr.set_xticks(np.linspace(0, data.shape[-1], data.shape[-1]//250))
    tfr.set_xticklabels(np.linspace(-(pre_win*1000), post_win*1000, data.shape[-1]//250))
    tfr.set_xlabel('Time (ms)', fontsize=12)
    tfr.set_ylabel('Frequency (Hz)', fontsize=12)
    tfr.vlines((pre_win * sr), 0, len(freqs)-1, 'k')

    f.suptitle(f'{title}')
    f.tight_layout()

    return f


def _rolling_rms_last_axis(data: np.ndarray, window_samples: int) -> np.ndarray:
    """Compute a trailing-window RMS along the last axis with min_periods=1 semantics."""

    if window_samples < 1:
        raise ValueError("window_samples must be at least 1")

    squared = np.square(np.asarray(data, dtype=_WORKING_DTYPE))
    cumulative = np.cumsum(squared, axis=-1, dtype=_WORKING_DTYPE)
    rolling_sum = cumulative
    if window_samples < squared.shape[-1]:
        # In-place update of the trailing window without a full buffer copy.
        rolling_sum = cumulative.copy()
        rolling_sum[..., window_samples:] = (
            cumulative[..., window_samples:] - cumulative[..., :-window_samples]
        )

    window_denominator = np.minimum(
        np.arange(1, squared.shape[-1] + 1, dtype=_WORKING_DTYPE),
        _WORKING_DTYPE(window_samples),
    )
    return np.sqrt(rolling_sum / window_denominator)


def _find_segments_above_threshold(mask: np.ndarray, min_length_samples: float, sfreq: float):
    """Convert a 1D boolean mask into start/stop/duration tuples."""

    padded_mask = np.pad(mask.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded_mask) == 1)
    stops = np.flatnonzero(np.diff(padded_mask) == -1)
    lengths = stops - starts
    valid = lengths > min_length_samples
    return list(zip(starts[valid], stops[valid], lengths[valid] / sfreq))

def detect_fast_burst_evs(mne_data, baseline_data, burst_frequency: tuple = (70, 200), smooth_win_s: float = 0.02, sd_upper_cutoff: float = 6, sd_lower_cutoff: float = 1, n_jobs: int = 1):
    """Detect fast burst events in HFA band (advanced / lightly maintained).

    Parameters
    ----------
    mne_data
        MNE epochs object.
    baseline_data
        Baseline MNE epochs object.
    burst_frequency : tuple, optional
        Frequency range for burst detection. Default is (70, 200).
    smooth_win_s : float, optional
        Smoothing window in seconds. Default is 0.02.
    sd_upper_cutoff : float, optional
        Upper SD cutoff. Default is 6.
    sd_lower_cutoff : float, optional
        Lower SD cutoff. Default is 1.
    n_jobs : int, optional
        Parallel jobs for filtering. Default is 1 (local-machine friendly).
        Pass ``n_jobs=-1`` on a cluster.

    Returns
    -------
    dict
        Dictionary of burst events per channel.
    """
    warnings.warn(
        "`detect_fast_burst_evs` is lightly maintained and may move to "
        "`LFPAnalysis._scratch_utils` in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )

    # set minimum burst duration to 3 cycles of the lower bound frequency
    min_burst_s = 3 / burst_frequency[0]

    # Step 1: band-pass filter the data
    from .mne_compat import filter_mne_object

    filtered_data = filter_mne_object(mne_data, burst_frequency[0], burst_frequency[1], n_jobs=n_jobs)

    smooth_win_samples = max(1, round(smooth_win_s * mne_data.info['sfreq']))
    rolling_rms_array = _rolling_rms_last_axis(filtered_data._data, smooth_win_samples)

    # Step 2: band-pass filter the baseline data
    filtered_baseline = filter_mne_object(
        baseline_data, burst_frequency[0], burst_frequency[1], n_jobs=n_jobs
    )

    rolling_rms_baseline = _rolling_rms_last_axis(filtered_baseline._data, smooth_win_samples)

    # calculate mean and standard deviation of smoothed rms data, across all epochs and timepoints
    smoothed_mean = rolling_rms_baseline.mean()
    smoothed_sd = rolling_rms_baseline.std()

    # calculate lower and upper cutoffs for marking burst events 
    lower_cutoff = smoothed_mean + sd_lower_cutoff * smoothed_sd
    upper_cutoff = smoothed_mean + sd_upper_cutoff * smoothed_sd

    burst_mask = (rolling_rms_array > lower_cutoff) & (rolling_rms_array < upper_cutoff)

    # Step 4: detected burst events with a duration shorter than 3 cycles of the lower bound frequency or longer than 500 ms, were rejected.
    min_length_burst_event = min_burst_s * mne_data.info['sfreq'] # add an input parameter for duration of burst event

    burst_samps_dict = {f'{x}':np.nan for x in mne_data.ch_names}
    event_indices = np.unique(np.where(burst_mask)[0])
    channel_indices = np.unique(np.where(burst_mask)[1])

    for ch_idx in channel_indices:
        burst_dict = {event_idx: np.nan for event_idx in event_indices}
        for event_idx in event_indices:
            burst_dict[event_idx] = _find_segments_above_threshold(
                burst_mask[event_idx, ch_idx, :],
                min_length_burst_event,
                mne_data.info['sfreq'],
            )
        burst_samps_dict[mne_data.ch_names[ch_idx]] = burst_dict

        # if plot:
        # # plot the burst using span for each trial and channel
        #     for ch in mne_data.ch_names:
        #         ch_index = mne_data.ch_names.index(ch)
        #         for trial in range(mne_data._data.shape[0]):
        #             plt.plot(rolling_rms_array[trial, ch_index, :], color='black')
        #             plt.plot(filtered_data._data[trial, ch_index, :], color='black', alpha=0.1)
        #             for start, stop, _ in hfa_bursts[ch][trial]:
        #                 plt.axvspan(start, stop, color='red', alpha=0.5)

    return burst_samps_dict

    
#     Foster et al., 
#     1. band-pass filtered from 80 to 120 Hz (ripple band) using a 4th order FIR filter.
#     2. the root mean square (RMS) of the band-passed signal was calculated and smoothed using a 20-ms window
#     3. ripple events were identified as having an RMS amplitude above 2.5, but no greater than 9, standard deviations from the mean
#     4. detected ripple events with a duration shorter than 38 ms (corresponding to 3 cycles at 80 Hz) or longer than 500 ms, were rejected.
  
#     """

#     # # What type of data is this? Continuous or epoched? 
#     # if type(mne_data) == mne.epochs.Epochs:
#     #     data_type = 'epoch'
#     # elif type(mne_data) == mne.io.fiff.raw.Raw: 
#     #     # , mne.io.edf.edf.RawEDF - probably should never include EDF data directly here. 
#     #     data_type = 'continuous'
#     # else: 
#     #     data_type = 'continuous'

#     # Step 1: band-pass filter from 80 - 120 Hz (ripple band) 
#     min_width = width_thresh * sr

#     # filter data in HFA band
#     filtered_data = mne_data.copy().filter(70, 200, n_jobs=-1)

#     # Create an empty array to store the rolling RMS for each trial and time series
#     rolling_rms_array = np.zeros_like(filtered_data._data)

#     # Loop over each trial and each time series and calculate the rolling RMS
#     for i in range(filtered_data._data.shape[0]):
#         column_values = ['signal'] 
#         df = pd.DataFrame(data = filtered_data._data[i, 0, :], columns = column_values)
#         smoothed_data = df['signal'].pow(2).rolling(round(smoothing_window_length * mne_data.info['sfreq']), min_periods=1).mean().apply(np.sqrt)
#         rolling_rms_array[i, :] = smoothed_data.values

#     # Step 3: mark ripple events [ripple start, ripple end] as periods of RMS amplitude above 2.5, but no greater than 9, standard deviations from the mean 

#     # calculate mean and standard deviation of smoothed rms data, across all epochs and timepoints
#     smoothed_mean = rolling_rms_array.mean()
#     smoothed_sd = rolling_rms_array.std()

#     # calculate lower (above 2.5 SD from mean) and upper (lower than 9 SD from mean) cutoffs for marking ripple events 
#     lower_cutoff = smoothed_mean + sd_lower_cutoff * smoothed_sd
#     upper_cutoff = smoothed_mean + sd_upper_cutoff * smoothed_sd

#     ripple_events_index = np.asarray(np.where((rolling_rms_array > lower_cutoff) & (rolling_rms_array < upper_cutoff)))

#     # Step 4: detected ripple events with a duration shorter than 38 ms (corresponding to 3 cycles at 80 Hz) or longer than 500 ms, were rejected.
#     min_length_ripple_event = min_ripple_length * mne_data.info['sfreq'] # add an input parameter for duration of ripple event
#     max_length_ripple_event = max_ripple_length * mne_data.info['sfreq']

#     RPL_samps_dict = {f'{x}':np.nan for x in mne_data.ch_names}
#     RPL_sec_dict = {f'{x}':np.nan for x in mne_data.ch_names}

#     for ch_ in np.unique(ripple_events_index[1]):
#         RPL_dict = {x:np.nan for x in np.unique(ripple_events_index[0])}
#         for ev in np.unique(ripple_events_index[0]):
#             # let's index the ripples for this ch_ and this ev 
#             ev_index = np.where(ripple_events_index[0] == ev)
#             ch_index = np.where(ripple_events_index[1] == ch_)
#             overlapping_index = np.intersect1d(ev_index, ch_index)
#             ripple_ch_ev = ripple_events_index[-1][overlapping_index]

#             ripple_events_differences = np.array([0] + np.diff(ripple_ch_ev))

#             # get the lengths and indices of consecutive 1s (this is how we know that they are sequential samples)
#             _, idx, counts = np.unique(np.cumsum(1-ripple_events_differences)*ripple_events_differences, return_index=True, return_counts=True)    

#             ripple_events_index_correct_time = idx[np.where((counts > min_length_ripple_event) & (counts < max_length_ripple_event))] # index of ripple events that reach criterion
#             ripple_events_length_samples = counts[np.where((counts > min_length_ripple_event) & (counts < max_length_ripple_event))]  # length in samples of ripple events that reach criterion
#             ripple_end_index = ripple_events_index_correct_time + ripple_events_length_samples
#             ripple_events_length_seconds = ripple_events_length_samples/mne_data.info['sfreq'] # length in seconds of ripple events that reach criterion

#             # # zip the three lists using zip() function --> ripple_results is a list of tuples containing the starting index of each ripple, the ending index of each ripple, and the length of each ripple in seconds
#             ripple_results = list(zip(ripple_ch_ev[ripple_events_index_correct_time],
#                                     ripple_ch_ev[ripple_end_index],
#                                     ripple_events_length_seconds))
#             num_ripples = len(ripple_results) # this is the number of ripples
#             RPL_dict[ev] = ripple_results
#         RPL_samps_dict[mne_data.ch_names[ch_]]= RPL_dict


#         # NOTE: you could TECHNICALLY stop here. However, a lot of these ripples are going to be 
#         # artifactual sharp transients that cover a lot of frequency range. So the next function is useful
#         # to look at the TFRs for each ripple. 

#     return RPL_samps_dict, RPL_sec_dict


#     Here we follow up our ripple detection with steps to filter for ripple events with specific spectrotemporal characteristics: 



def FOOOF_continuous(signal: np.ndarray):
    """Archived stub — continuous FOOOF was never implemented."""
    from ._scratch_utils import FOOOF_continuous as _archived

    return _archived(signal)


def FOOOF_compute_epochs(epochs, tmin: float = 0, tmax: float = 1.5, **kwargs):
    """Compute FOOOF on epoched data.

    Prefer ``workflow.compute_spectral_features`` / ``build_spectral_pipeline_config``
    for beginner workflows. Requires the ``fooof`` extra
    (``pip install -e '.[analysis]'``).

    Parameters
    ----------
    epochs
        MNE Epochs object.
    tmin : float, optional
        Start time in seconds. Default is 0.
    tmax : float, optional
        End time in seconds. Default is 1.5.
    **kwargs
        Required FOOOFGroup keys: ``peak_width_limits``, ``min_peak_height``,
        ``peak_threshold``, ``max_n_peaks``, ``freq_range``.

    Returns
    -------
    tuple
        Tuple containing (FOOOFGroup_res, pd.DataFrame).
    """
    fooof = ensure_dependency("fooof", install_hint="pip install -e '.[analysis]'")
    from fooof import FOOOFGroup

    epo_spectrum = epochs.compute_psd(method='multitaper',
                                                tmin=tmin,
                                                tmax=tmax,
                                                verbose=False)
                                                
    psds = np.asarray(epo_spectrum._data, dtype=_WORKING_DTYPE)
    freqs = epo_spectrum.freqs
            
    # average across epochs
    psd_trial_avg = np.nanmean(psds, axis=0)

    required = (
        'peak_width_limits',
        'min_peak_height',
        'peak_threshold',
        'max_n_peaks',
        'freq_range',
    )
    missing = [k for k in required if k not in kwargs]
    if missing:
        raise ValueError(f"FOOOF_compute_epochs missing required kwargs: {missing}")

    # Initialize a FOOOFGroup object, with desired settings
    FOOOFGroup_res = FOOOFGroup(peak_width_limits=kwargs['peak_width_limits'], 
                    min_peak_height=kwargs['min_peak_height'],
                    peak_threshold=kwargs['peak_threshold'], 
                    max_n_peaks=kwargs['max_n_peaks'], 
                    verbose=False)

    # Fit the FOOOF object 
    FOOOFGroup_res.fit(freqs, psd_trial_avg, kwargs['freq_range'])

    all_chan_dfs = []
    # Go through individual channels; build DataFrames from arrays (avoid .tolist() copies).
    for chan in range(len(epochs.ch_names)):

        ind_fits = FOOOFGroup_res.get_fooof(ind=chan, regenerate=True)
        ind_fits.fit()
        # FOOOF trims to freq_range; peak markers must match that length (not full PSD).
        n_freqs = len(ind_fits.freqs)

        # Create a dataframe to store results 
        chan_data_df = pd.DataFrame({
            'channel': epochs.ch_names[chan],
            'frequency': ind_fits.freqs,
            'PSD_raw': ind_fits.power_spectrum,
            'PSD_corrected': ind_fits._spectrum_flat,
            'in_FOOOF_peak': np.zeros(n_freqs, dtype=_WORKING_DTYPE),
            'peak_freq': np.zeros(n_freqs, dtype=_WORKING_DTYPE),
            'peak_height': np.zeros(n_freqs, dtype=_WORKING_DTYPE),
            'PSD_exp': ind_fits.get_params('aperiodic_params', 'exponent'),
        })

        # Get peak info
        peaks = fooof.analysis.get_band_peak_fm(ind_fits, band=(1, 30), select_highest=False)
        in_FOOOF_peaks = np.zeros(n_freqs, dtype=_WORKING_DTYPE)
        peak_freqs = np.zeros(n_freqs, dtype=_WORKING_DTYPE)
        peak_heights = np.zeros(n_freqs, dtype=_WORKING_DTYPE)
        
        # Iterate through the peaks and create dataframe friendly data that assigns each frequency to a peak (or not)
        if peaks is not None and np.ndim(peaks) == 1: # only one peak

            center_pk = peaks[0]
            low_freq = peaks[0] - (peaks[2]/2)
            high_freq = peaks[0] + (peaks[2]/2)
            pk_height = peaks[1]
            mask = (ind_fits.freqs >= low_freq) & (ind_fits.freqs <= high_freq)
            in_FOOOF_peaks[mask] = 1
            peak_freqs[mask] = center_pk
            peak_heights[mask] = pk_height

        elif peaks is not None and np.ndim(peaks) > 1: # more than one peak
            for ix, pk in enumerate(peaks):
                center_pk = pk[0]
                low_freq = pk[0] - (pk[2]/2)
                high_freq = pk[0] + (pk[2]/2)
                pk_height = pk[1]
                mask = (ind_fits.freqs >= low_freq) & (ind_fits.freqs <= high_freq)
                in_FOOOF_peaks[mask] = ix + 1
                peak_freqs[mask] = center_pk
                peak_heights[mask] = pk_height

        chan_data_df['in_FOOOF_peak'] = in_FOOOF_peaks
        chan_data_df['peak_freq'] = peak_freqs
        chan_data_df['peak_height'] = peak_heights

        all_chan_dfs.append(chan_data_df)


    return FOOOFGroup_res, pd.concat(all_chan_dfs)



#         cond_df['condition'] = cond

#         all_cond_df.append(cond_df)

#     # Go through individual channels
#     for chan in range(len(epochs_with_metadata.ch_names)):
#         file_name = f'{epochs_with_metadata.ch_names[chan]}_PSD'

#         cond_fits = [fooof_groups_cond[cond].get_fooof(ind=chan, regenerate=True) for cond in conditions]
#         for i in range(len(cond_fits)):
#             cond_fits[i].fit()

#         # Create a dataframe to store results 
#         chan_data_df = pd.DataFrame(columns=['exp_diff', 'peak_pow_diff', 'band_pow_diff', 'band_pow_diff_flat', 'band'])

#         # Compute contrast between conditions
#         exp_diff = _compare_exp(cond_fits[0], cond_fits[1])

#         band_labels = []
#         peak_pow_diffs = [] 
#         band_pow_diffs = []
#         band_pow_diff_flats = []

#         for label, definition in bands:
#             band_labels.append(label)
#             peak_pow_diffs.append(_compare_peak_pw(cond_fits[0], cond_fits[1], definition))
#             band_pow_diffs.append(_compare_band_pw(cond_fits[0], cond_fits[1], definition))
#             band_pow_diff_flats.append(_compare_band_pw_flat(cond_fits[0], cond_fits[1], definition))

#         chan_data_df['peak_pow_diff'] = peak_pow_diffs
#         chan_data_df['band_pow_diff'] = band_pow_diffs
#         chan_data_df['band_pow_diff_flat'] = band_pow_diff_flats
#         chan_data_df['band'] = band_labels
#         chan_data_df['exp_diff'] = exp_diff
#         chan_data_df['channel'] = epochs_with_metadata.ch_names[chan]
#         chan_data_df['region'] = epochs_with_metadata.metadata.region.unique()[0]

#         all_chan_dfs.append(chan_data_df)

#         if plot: 
#             with PdfPages(f'{file_path}/{file_name}.pdf') as pdf:
#                 f, ax = plt.subplots(1, 2, figsize=[18, 6], dpi=300)
#                 # Plot the power spectra differences, representing the 'band-by-band' idea
#                 fooof.plts.spectra.plot_spectra_shading(cond_fits[0].freqs, 
#                                                         [x.power_spectrum for x in cond_fits],
#                                                         log_powers=False, linewidth=3,
#                                                         shades=bands.definitions, shade_colors=shade_cols,
#                                                         labels=conditions,
#                                                         ax=ax[0])
#                 ax[0].set_title(f'{epochs_with_metadata.ch_names[chan]}', t_settings)

#                 # Plot the flattened power spectra differences
#                 fooof.plts.spectra.plot_spectra_shading(cond_fits[0].freqs, 
#                                                         [x._spectrum_flat for x in cond_fits],
#                                                         log_powers=False, linewidth=3,
#                                                         shades=bands.definitions, shade_colors=shade_cols,
#                                                         labels=conditions,
#                                                         ax=ax[1])

#                 ax[1].set_title(f'{epochs_with_metadata.ch_names[chan]} - flattened')

#                 f.tight_layout()

#                 pdf.savefig()
#                 plt.close(f)

#     return pd.concat(all_chan_dfs), pd.concat(all_cond_df)

# We put all of our basic FOOOF usage into a slightly clunky function that is meant to be used for running the regression
# over multiple channels in parallel using joblib/Dask/multiprocessing.Pool: 
def compute_FOOOF_parallel(chan_name: str, MNE_object, subj_id: str, elec_df: pd.DataFrame, event_name: str, ev_dict: dict, band_dict: dict, conditions: list, do_plot: bool = False, save_path: str | None = None, do_save: bool = False, **kwargs):
    """Compute FOOOF for single channel in parallel.
    
    Parameters
    ----------
    chan_name : str
        Channel name.
    MNE_object
        MNE Epochs object.
    subj_id : str
        Subject ID.
    elec_df : pd.DataFrame
        Electrode DataFrame.
    event_name : str
        Event name.
    ev_dict : dict
        Event time dictionary.
    band_dict : dict
        Frequency band dictionary.
    conditions : list
        List of conditions.
    do_plot : bool, optional
        Whether to plot. Default is False.
    save_path : str, optional
        Root directory for optional CSV/plot outputs. Default is ``None``
        (required when ``do_save=True``). Cluster paths are no longer hard-coded.
    do_save : bool, optional
        Whether to save. Default is False.
    **kwargs
        Additional FOOOF arguments.

    Returns
    -------
    pd.DataFrame or None
        Results DataFrame if not saving.
    """
    if do_save and not save_path:
        raise ValueError("save_path is required when do_save=True")

    dfs = []
    # Can pick the epoch depending on the event being selected
    chan_epochs = MNE_object.copy().pick([chan_name])

    # FOOOF across all trials: 
    FOOOFGroup_res, df_all = FOOOF_compute_epochs(chan_epochs, tmin=ev_dict[event_name][0], tmax=ev_dict[event_name][1], 
                                                        band_dict=band_dict, **kwargs)

    df_all['PSD_raw'] =  sp.stats.zscore(df_all['PSD_raw'])
    # df_all['PSD_corrected'] =  sp.stats.zscore(df_all['PSD_corrected'])
    df_all['cond'] = 'all'
    df_all['event'] = event_name
    df_all['region'] = elec_df[elec_df.label==chan_name].salman_region.values[0]

    dfs.append(df_all)

    # Second, compute FOOOF only for the trials belonging to each condition of interest
    df_conds = []
    for cond in conditions: 

        chan_epochs = MNE_object[cond].copy().pick([chan_name])

        FOOOFGroup_res, df_temp = FOOOF_compute_epochs(chan_epochs, tmin=ev_dict[event_name][0], tmax=ev_dict[event_name][1], 
                                                        band_dict=band_dict, **kwargs)

        df_temp['cond'] = cond
        df_temp['event'] = event_name

        df_temp['region'] = elec_df[elec_df.label==chan_name].salman_region.values[0]

        df_conds.append(df_temp)

    df_conds = pd.concat(df_conds)
    df_conds['PSD_raw'] =  sp.stats.zscore(df_conds['PSD_raw'])
    # df_conds['PSD_corrected'] =  sp.stats.zscore(df_conds['PSD_corrected'])
    dfs.append(df_conds)

    chan_df = pd.concat(dfs)
    chan_df.insert(0,'participant', subj_id)

    if do_plot:
        fig = sns.lineplot(data=chan_df, x='frequency', y='PSD_corrected', hue='cond')
        figure = fig.get_figure()    
        figure.savefig(f'{save_path}/{subj_id}/scratch/FOOOF/{event_name}/plots/{chan_name}_FOOOF.pdf', dpi=100)
        plt.close()

    if do_save:
        # save this chan_df out 
        chan_df.to_csv(f'{save_path}/{subj_id}/scratch/FOOOF/{event_name}/dfs/{chan_name}_df.csv', index=False)
    else:
        return chan_df


def sliding_FOOOF(signal: np.ndarray):
    """Archived stub — sliding FOOOF was never implemented."""
    from ._scratch_utils import sliding_FOOOF as _archived

    return _archived(signal)


def hctsa_signal_features(signal: np.ndarray):
    """Archived catch22 wrapper — prefer calling ``pycatch22`` directly."""
    from ._scratch_utils import hctsa_signal_features as _archived

    return _archived(signal)
