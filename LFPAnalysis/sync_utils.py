import numpy as np
import scipy.stats
import warnings
from scipy.stats import spearmanr 

# Utility functions for synchronization

# Might be nice to synergize with https://github.com/alexrockhill/pd-parser to see if there's some improvements to be made

def get_behav_ts(logfile): 
    """
    Insert custom function to extract the behavioral timestamps from the logfile for your task. 
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

def pulsealign(beh_ms=None,
               pulses=None, 
               windSize=15):
    """
    Aligns the behavioral timestamps with the EEG pulses by finding the chunks of behavioral pulse times where the inter-pulse intervals are correlated with the EEG pulses.

    Parameters
    ----------
    beh_ms (np.ndarray): A vector of ms times extracted from the log file.
    pulses (np.ndarray): Vector of EEG pulses extracted from the EEG.
    windSize (int): The size of the chunks to step through the recorded sync pulses. Default is 15.

    Returns
    -------
    A tuple of two np.ndarrays:
        - beh_ms: The truncated beh_ms values that match the eeg_offset.
        - eeg_offset: The truncated pulses that match the beh_ms.
    """
    
    # # THIS SHIT RETURNS > 1 SOMETIMES??? CHECK ZE MATH
    def fastCorr(x, y):
        # faster version of corr
        c = np.cov(x, y)
        r = c[0, 1] / (np.std(x) * np.std(y))
        return r

    # these are parameters that one could potentially tweak....
    corrThresh = 0.99
    
    eegBlockStart = np.arange(0, len(pulses) - windSize + 1, windSize)
    
    beh_d = np.diff(beh_ms)
    # beh_d[beh_d > 20*1000] = 0  # if any interpulse differences are greater than twenty seconds, throw them out!
    pulse_d = np.diff(pulses)
    
    print(f"{len(eegBlockStart)} blocks")
    
    blockR = np.zeros(len(eegBlockStart))
    blockBehMatch = np.zeros(len(eegBlockStart), dtype=int)
    
    for b in range(len(eegBlockStart)):
        print(".", end="")
        eeg_d = pulse_d[eegBlockStart[b]:eegBlockStart[b]+windSize]
        r = np.zeros(len(beh_d) - len(eeg_d))
        p = np.zeros(len(beh_d) - len(eeg_d))
        for i in range(len(beh_d) - len(eeg_d)):
            # sometimes the lengths mismatch by one entry if we are by an edge: 
            length = min(len(eeg_d), len(beh_d[i:i+windSize]))
            r[i] = fastCorr(eeg_d[:length], beh_d[i:i+length])
            if r[i] > 1: 
                # failure mode
                res = spearmanr(eeg_d[:length], beh_d[i:i+length])
                r[i] = res[0]
            # r[i] = fastCorr(eeg_d, beh_d[i:i+windSize])
        blockR[b] = np.max(r)
        blockBehMatch[b] = np.argmax(r)
    print("\n")
    
    # now, for each block, check if it had a good correlation. if so, then add the set of matching pulses into the output
    
    eeg_offset = np.array([])
    good_beh_ms = np.array([])
    
    for b in np.where(blockR > corrThresh)[0]:
        x = pulses[eegBlockStart[b]:eegBlockStart[b]+windSize]
        eeg_offset = np.concatenate([eeg_offset, x])
        y = beh_ms[blockBehMatch[b]:blockBehMatch[b]+windSize]
        good_beh_ms = np.concatenate([good_beh_ms, y])
    
    print(f"found matches for {len(eeg_offset)} of {len(pulses)} pulses")
    
    return good_beh_ms, eeg_offset

def sync_matched_pulses(beh_pulse, neural_pulse):
    """
    Compute the slope and offset of the linear regression between two sets of pulse timestamps.

    Parameters:
    beh_pulse (array-like): The timestamps of the behavioral pulses.
    neural_pulse (array-like): The timestamps of the neural pulses.

    Returns:
    tuple: A tuple containing the slope, offset, and correlation coefficient of the linear regression.

    Note:     Idea is similar to this: https://github.com/mne-tools/mne-python/blob/main/mne/preprocessing/realign.py#L13-L111

    """
    bfix = beh_pulse[0]
    res = scipy.stats.linregress(beh_pulse-bfix, neural_pulse)
    slope = res[0]
    offset = res[1]
    offset = offset - bfix*slope
    rval = res[2]

    return slope, offset, rval


def synchronize_data(beh_ts, mne_sync, smoothSize=11, windSize=15, height=0.5):
    """
    Synchronize the behavioral timestamps from the logfile and the mne photodiode data and return the slope and offset for the session.

    Parameters:
    beh_ts (array-like): The timestamps of the behavioral events.
    mne_sync (MNE object): The MNE photodiode data.
    smoothSize (int): The size of the smoothing window for the photodiode data.
    windSize (int): The size of the window for pulse alignment.
    height (float): The threshold for detecting the rising edge of the photodiode signal.

    Returns:
    tuple: A tuple containing the slope and offset of the linear regression between the behavioral and neural timestamps.

    Raises:
    ValueError: If the synchronization fails.

    Note: 
    - The function uses a moving average filter to smooth the photodiode data.
    - The function uses a z-score normalization for the photodiode data.
    - The function uses a threshold to detect the rising edge of the photodiode signal.
    - The function uses pulse alignment to match the behavioral and neural timestamps.
    - The function uses linear regression to compute the slope and offset of the synchronization.
    - The function increases the window size for pulse alignment until the correlation coefficient of the linear regression is greater than or equal to 0.99.
    - The function raises a ValueError if the synchronization fails.
    """
    
    sig = np.squeeze(moving_average(mne_sync._data, n=smoothSize))
    timestamp = np.squeeze(np.arange(len(sig))/mne_sync.info['sfreq'])
    sig = scipy.stats.zscore(sig)

    trig_ix = np.where((sig[:-1]<=height)*(sig[1:]>height))[0] # rising edge of trigger
    
    neural_ts = timestamp[trig_ix]
    neural_ts = np.array(neural_ts)

    if len(neural_ts) < (len(beh_ts)//1.5): 
        warnings.warn("Your height parameter may be too strict - consider setting it a little lower")

    if len(neural_ts) > (len(beh_ts)*1.5): 
        warnings.warn("Your height parameter may be too lenient - consider setting it a little higher")

    rval = 0 
    while (rval<0.99) & (windSize < 60):
        if len(beh_ts)!=len(neural_ts):
            good_beh_ts, good_neural_ts = pulsealign(beh_ts, neural_ts, windSize=windSize)
            slope, offset, rval = sync_matched_pulses(good_beh_ts, good_neural_ts)
        else:
            slope, offset, rval = sync_matched_pulses(beh_ts, neural_ts)
        windSize += 5
    if rval < 0.99:
        raise ValueError(f'this sync for subject has failed - examine the data')
    else:
        return slope, offset


