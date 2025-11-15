import numpy as np
import scipy.stats
import warnings
from scipy.stats import pearsonr
from collections import defaultdict 

# Utility functions for synchronization

# Might be nice to synergize with https://github.com/alexrockhill/pd-parser to see if there's some improvements to be made

def get_behav_ts(logfile):
    """Extract behavioral timestamps from logfile.
    
    Parameters
    ----------
    logfile
        Logfile to extract timestamps from.
    """
    pass
    

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

# def fastCorr(x, y):
#     # faster version of corr
#     # # THIS SHIT RETURNS > 1 SOMETIMES??? CHECK ZE MATH

#     c = np.cov(x, y)
#     r = c[0, 0] / (np.std(x) * np.std(y))
#     return r

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
    
    print(f"{len(eegBlockStart)} blocks")
    
    blockR = np.zeros(len(eegBlockStart))
    blockBehMatch = np.zeros(len(eegBlockStart), dtype=int)

    # iterate through blocks of neural ts
    for b in range(len(eegBlockStart)):
        print(".", end="")
        eeg_d = pulse_d[eegBlockStart[b]:eegBlockStart[b]+windSize]
        r = np.zeros(len(beh_d) - len(eeg_d))
        p = np.zeros(len(beh_d) - len(eeg_d))
        for i in range(len(beh_d) - len(eeg_d)):
            # sometimes the lengths mismatch by one entry if we are by an edge: 
            length = min(len(eeg_d), len(beh_d[i:i+windSize]))
            # 1/31/24: slows down correlation but more confident in result.
            r[i] = np.corrcoef(beh_d[i:i+length], eeg_d[:length])[0, 1]

        blockR[b] = np.max(r)
        blockBehMatch[b] = np.argmax(r)
    print("\n")
    
    # now, for each block, check if it had a good correlation. if so, then add the set of matching pulses into the output
    
    eeg_offset = np.array([])
    good_beh_ms = np.array([])
    
    for b in np.where(blockR > corrThresh)[0]:
        x = pulses[eegBlockStart[b]:eegBlockStart[b]+windSize]
        y = beh_ms[blockBehMatch[b]:blockBehMatch[b]+windSize]
        slope, offset, rval = sync_matched_pulses(y, x)
        # 1/31/24: Let's only concatenate if slope is within some reasonable distance to 1
        if (rval > corrThresh) & (np.abs(1-slope)<=0.05):
            eeg_offset = np.concatenate([eeg_offset, x])
            
            good_beh_ms = np.concatenate([good_beh_ms, y])
            # FOR DEBUGGING:
            # print(slope)
            # print(offset)
            # print(rval)
        
    print(f"found matches for {len(eeg_offset)} of {len(pulses)} pulses")
    
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

    # Initialize variables to store matching epochs
    matching_epochs = []

    corr = -1
    # Iterate through windows in neural_diff
    for i in range(0, len(neural_diff) - window_size + 1, step_size):
        print(".", end="")
        neural_window = neural_diff[i:i + window_size]
        if corr > correlation_threshold:
            continue
        # Iterate through windows in beh_diff
        for j in range(0, len(beh_diff) - window_size + 1, step_size):
            beh_window = beh_diff[j:j + window_size]

            # Calculate Pearson correlation coefficient
            corr, _ = pearsonr(beh_window, neural_window)

            # Check if correlation coefficient exceeds the threshold
            if corr > correlation_threshold:
                # Save matching epoch details
                neural_matching_window = neural_ts[i:i + window_size + 1]
                beh_matching_window = beh_ts[j:j + window_size + 1]
                slope, offset, rval = sync_matched_pulses(beh_matching_window, neural_matching_window)
                if np.abs(1-slope)<=0.05:
                    matching_epochs.append({
                        'neural_timestamps': neural_matching_window,
                        'beh_timestamps': beh_matching_window,
                        'slope': slope,
                        'offset': offset,
                        'correlation_coefficient': rval
                    })
        corr = -1
    print("\n")

    merged_dict = defaultdict(list)

    for d in matching_epochs:
        for key, value in d.items():
            merged_dict[key].append(value)

    # If you want to convert defaultdict back to a regular dictionary
    merged_dict = dict(merged_dict)
    
    # stack and compute final sync
    slope, offset, rval = sync_matched_pulses(np.hstack(merged_dict['beh_timestamps']), 
                                          np.hstack(merged_dict['neural_timestamps']))      

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
    except: 
        print('fast sync failed - running robust sync now')
        while (rval<0.99) & (windSize < 60):
            windSize = 15
            slope, offset, rval = synchronize_data_robust(beh_ts, neural_ts, window_size=windSize, step_size=1)
            windSize += 5
        if rval < 0.99:
            raise ValueError(f'this sync for subject has failed - CHECK YOUR INPUT DATA')
        else:
            print('successful sync!')
    return slope, offset

