import numpy as np
import scipy.stats
import warnings

# Utility functions for synchronization

# Might be nice to synergize with https://github.com/alexrockhill/pd-parser to see if there's some improvements to be made

def get_behav_ts(logfile):
    """Archived stub — extract behavioral timestamps yourself, then sync.

    Parameters
    ----------
    logfile
        Logfile to extract timestamps from.
    """
    from ._scratch_utils import get_behav_ts as _archived

    return _archived(logfile)


def moving_average(a, n=11) :
    """
    Computes the moving average of a given array a with a window size of n.

    Parameters
    ----------
    a : np.ndarray
        The input array to compute the moving average on.
    n : int, optional
        The window size of the moving average. Default is 11.

    Returns
    -------
    np.ndarray
        The moving average of the input array a.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def _normalized_sliding_windows(data, window_size: int, step_size: int = 1) -> np.ndarray:
    """Return z-normalized sliding windows for vectorized correlation."""

    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(data, dtype=float),
        window_shape=window_size,
    )[::step_size]
    centered = windows - windows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)

def get_neural_ts_photodiode(mne_sync, smoothSize: int = 11, height: float = 0.5):
    """Extract neural timestamps from photodiode signal.
    
    Parameters
    ----------
    mne_sync
        MNE sync data object.
    smoothSize : int, optional
        Smoothing window size. Default is 11.
    height : float, optional
        Threshold for detecting rising edge. Default is 0.5.
    
    Returns
    -------
    np.ndarray
        Neural timestamps.
    """

    sig = np.squeeze(moving_average(mne_sync._data, n=smoothSize))
    timestamp = np.squeeze(np.arange(len(sig))/mne_sync.info['sfreq'])
    sig = scipy.stats.zscore(sig)

    trig_ix = np.where((sig[:-1]<=height)*(sig[1:]>height))[0] # rising edge of trigger
    
    neural_ts = timestamp[trig_ix]
    neural_ts = np.array(neural_ts)

    return neural_ts

def get_neural_ts_ttl(nev_data):
    """Extract neural timestamps from TTL recording.
    
    Parameters
    ----------
    nev_data : dict
        NEV data dictionary containing records.
    
    Returns
    -------
    np.ndarray
        Neural timestamps in seconds.
    """

    return nev_data['records']['TimeStamp'][nev_data['records']['ttl']==1] * 1e-6

def pulsealign(beh_ms=None, pulses=None, windSize: int = 15):
    """Align behavioral timestamps with EEG pulses.
    
    Parameters
    ----------
    beh_ms : np.ndarray, optional
        Vector of ms times extracted from the log file.
    pulses : np.ndarray, optional
        Vector of EEG pulses extracted from the EEG.
    windSize : int, optional
        Size of chunks to step through recorded sync pulses. Default is 15.
    
    Returns
    -------
    tuple
        Tuple of (beh_ms, eeg_offset) np.ndarrays.
    """

    # these are parameters that one could potentially tweak....
    corrThresh = 0.99
    
    eegBlockStart = np.arange(0, len(pulses) - windSize + 1, windSize)
    
    beh_d = np.diff(beh_ms)
    pulse_d = np.diff(pulses)

    if len(beh_d) < windSize or len(pulse_d) < windSize:
        return np.array([]), np.array([])

    beh_windows = _normalized_sliding_windows(beh_d, windSize)
    
    blockR = np.zeros(len(eegBlockStart))
    blockBehMatch = np.zeros(len(eegBlockStart), dtype=int)

    # iterate through blocks of neural ts
    for b in range(len(eegBlockStart)):
        eeg_d = pulse_d[eegBlockStart[b]:eegBlockStart[b]+windSize]
        eeg_centered = eeg_d - eeg_d.mean()
        eeg_norm = np.linalg.norm(eeg_centered)
        if eeg_norm == 0:
            blockR[b] = np.nan
            continue
        r = beh_windows @ (eeg_centered / eeg_norm)

        blockR[b] = np.max(r)
        blockBehMatch[b] = np.argmax(r)
    
    # now, for each block, check if it had a good correlation. if so, then add the set of matching pulses into the output
    
    matched_eeg_offsets = []
    matched_beh_ms = []
    
    for b in np.where(blockR > corrThresh)[0]:
        x = pulses[eegBlockStart[b]:eegBlockStart[b]+windSize]
        y = beh_ms[blockBehMatch[b]:blockBehMatch[b]+windSize]
        slope, offset, rval = sync_matched_pulses(y, x)
        # 1/31/24: Let's only concatenate if slope is within some reasonable distance to 1
        if (rval > corrThresh) & (np.abs(1-slope)<=0.05):
            matched_eeg_offsets.append(x)
            matched_beh_ms.append(y)

    if not matched_eeg_offsets:
        return np.array([]), np.array([])

    eeg_offset = np.concatenate(matched_eeg_offsets)
    good_beh_ms = np.concatenate(matched_beh_ms)
    
    return good_beh_ms, eeg_offset

def sync_matched_pulses(beh_pulse, neural_pulse):
    """Compute slope and offset of linear regression between pulse timestamps.
    
    Parameters
    ----------
    beh_pulse : array-like
        Timestamps of behavioral pulses.
    neural_pulse : array-like
        Timestamps of neural pulses.
    
    Returns
    -------
    tuple
        Tuple containing (slope, offset, rval).
    """
    bfix = beh_pulse[0]
    res = scipy.stats.linregress(beh_pulse-bfix, neural_pulse)
    slope = res[0]
    offset = res[1]
    offset = offset - bfix*slope
    rval = res[2]

    return slope, offset, rval

def synchronize_data_robust(beh_ts=None, neural_ts=None, window_size: int = 15, step_size: int = 1, correlation_threshold: float = 0.99):
    """Robustly synchronize behavioral and neural timestamps.
    
    Parameters
    ----------
    beh_ts : array-like, optional
        Behavioral timestamps.
    neural_ts : array-like, optional
        Neural timestamps.
    window_size : int, optional
        Window size for matching. Default is 15.
    step_size : int, optional
        Step size for iteration. Default is 1.
    correlation_threshold : float, optional
        Correlation threshold for matching. Default is 0.99.
    
    Returns
    -------
    tuple
        Tuple containing (slope, offset, rval).
    """
    # Calculate differences between consecutive timestamps
    neural_diff = np.diff(neural_ts)
    beh_diff = np.diff(beh_ts)

    if len(neural_diff) < window_size or len(beh_diff) < window_size:
        raise ValueError("Not enough timestamps to compute robust synchronization.")

    neural_windows = _normalized_sliding_windows(neural_diff, window_size, step_size)
    beh_windows = _normalized_sliding_windows(beh_diff, window_size, step_size)
    neural_starts = np.arange(0, len(neural_diff) - window_size + 1, step_size)
    beh_starts = np.arange(0, len(beh_diff) - window_size + 1, step_size)

    correlation_matrix = neural_windows @ beh_windows.T
    matching_pairs = np.argwhere(correlation_matrix > correlation_threshold)

    matched_neural = []
    matched_beh = []
    for neural_idx, beh_idx in matching_pairs:
        i = neural_starts[neural_idx]
        j = beh_starts[beh_idx]
        neural_matching_window = neural_ts[i:i + window_size + 1]
        beh_matching_window = beh_ts[j:j + window_size + 1]
        slope, offset, rval = sync_matched_pulses(beh_matching_window, neural_matching_window)
        if np.abs(1-slope)<=0.05:
            matched_neural.append(neural_matching_window)
            matched_beh.append(beh_matching_window)

    if not matched_neural:
        raise ValueError("No matching epochs exceeded the requested correlation threshold.")

    # stack and compute final sync
    slope, offset, rval = sync_matched_pulses(
        np.hstack(matched_beh),
        np.hstack(matched_neural),
    )

    return slope, offset, rval

def synchronize_data(beh_ts=None, mne_sync=None, smoothSize: int = 11, windSize: int = 15, height: float = 0.5, sync_source: str = 'photodiode'):
    """Synchronize behavioral timestamps with MNE photodiode data.
    
    Parameters
    ----------
    beh_ts : array-like, optional
        Timestamps of behavioral events.
    mne_sync
        MNE photodiode data or NEV data for TTL.
    smoothSize : int, optional
        Smoothing window size. Default is 11.
    windSize : int, optional
        Window size for pulse alignment. Default is 15.
    height : float, optional
        Threshold for detecting rising edge. Default is 0.5.
    sync_source : str, optional
        Type of signal used to sync data. Default is 'photodiode'.
    
    Returns
    -------
    tuple
        Tuple containing (slope, offset).
    
    Raises
    ------
    ValueError
        If synchronization fails.
    """

    if isinstance(sync_source, str):
        # This indicates I need to extract the syncs myself 
        if sync_source=='photodiode':
            neural_ts = get_neural_ts_photodiode(mne_sync, smoothSize, height)
            
            if len(neural_ts) < (len(beh_ts)//1.5): 
                warnings.warn("Your height parameter may be too strict - consider setting it a little lower")

            if len(neural_ts) > (len(beh_ts)*1.5): 
                warnings.warn("Your height parameter may be too lenient - consider setting it a little higher")

        elif sync_source=='ttl':
            neural_ts = get_neural_ts_ttl(mne_sync)
    elif isinstance(sync_source, np.ndarray) | isinstance(sync_source, list):
        # This indicates I am providing the extracted syncs myself
        neural_ts = sync_source

    rval = 0 
    try:
        while (rval<0.99) & (windSize < 60):
            if len(beh_ts)!=len(neural_ts):
                good_beh_ts, good_neural_ts = pulsealign(beh_ts, neural_ts, windSize=windSize)
                slope, offset, rval = sync_matched_pulses(good_beh_ts, good_neural_ts)
            else:
                slope, offset, rval = sync_matched_pulses(beh_ts, neural_ts)
            windSize += 5
        if rval < 0.99:
            raise ValueError(f'this sync for subject has failed - running robust synch now')
    except Exception:
        windSize = 15
        while (rval<0.99) & (windSize < 60):
            slope, offset, rval = synchronize_data_robust(beh_ts, neural_ts, window_size=windSize, step_size=1)
            windSize += 5
        if rval < 0.99:
            raise ValueError(f'this sync for subject has failed - CHECK YOUR INPUT DATA')
    return slope, offset
