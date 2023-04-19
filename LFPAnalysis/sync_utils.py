import numpy as np
import scipy.stats
import warnings

# Utility functions for synchronization

def get_behav_ts(logfile, format='old'): 
    """
    Gets the timestamps from the behavioral logfile depending on the format of the task (old? or new?)
    """
    pass
    

def moving_average(a, n=11) :
    """
    Clean up the sync channel a bit. 
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def pulsealign(beh_ms, pulses, windSize=30):

    # FUNCTION:
    #   function [beh_ms, eeg_offset] = pulsealign2(beh_ms, pulses, pulseIsMS)
    #
    # INPUT ARGS:
    #   beh_ms = beh_ms;   % A vector of ms times extracted from the
    #                      %  log file
    #   pulses = pulses;   % Vector of eeg pulses extracted from the eeg
    #
    # OUTPUT ARGS:
    #   beh_ms- The truncated beh_ms values that match the eeg_offset
    #   eeg_offset- The trucated pulses that match the beh_ms
    
    #  Step through the recorded sync pulses in chunks of  windsize.  Use corr to find the chunks of behavioral pulse times where the inter-pulse intervals are correlated.  When the maximum correlation is greater than corrThresh, then it indicates that the pairs match.
    
    # note that sampling rate never comes in here. this is how alignment should work---it should be entirely sampling-rate independent....
    
    
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
    Idea is similar to this: https://github.com/mne-tools/mne-python/blob/main/mne/preprocessing/realign.py#L13-L111
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

    Input the behavioral timestamps from the logfile and the mne photodiode data and return the slope and offset for the session.

    """

    sig = np.squeeze(moving_average(mne_sync._data, n=smoothSize))
    timestamp = np.squeeze(np.arange(len(sig))/mne_sync.info['sfreq'])
    sig = scipy.stats.zscore(sig)

    trig_ix = np.where((sig[:-1]<=height)*(sig[1:]>height))[0] # rising edge of trigger
    
    neural_ts = timestamp[trig_ix]
    neural_ts = np.array(neural_ts)

    # Optional warnings for height threshold to maximize success rate of finding a match
    if len(neural_ts) < (len(beh_ts)//2): 
        warnings.warn("Your height parameter may be too strict - consider setting it a little lower")

   if len(neural_ts) <>(len(beh_ts)*2): 
        warnings.warn("Your height parameter may be too lenient - consider setting it a little higher")


    rval = 0 

    while (rval<0.99) & (windSize < 60):
            if len(beh_ts)!=len(neural_ts):
                # Do regression to find neural timestamps for each event type
                good_beh_ts, good_neural_ts = pulsealign(beh_ts, neural_ts, windSize=windSize)
                slope, offset, rval = sync_matched_pulses(good_beh_ts, good_neural_ts)
            else:
                slope, offset, rval = sync_matched_pulses(beh_ts, neural_ts)
            windSize += 5

    if rval < 0.99:
        raise ValueError(f'this sync for subject has failed - examine the data')
    else:
        return slope, offset


