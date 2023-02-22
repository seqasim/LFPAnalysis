import numpy as np
import scipy.stats

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

def pulsealign(beh_ts, neural_ts, window=30, thresh=0.99):
    """
    Step through recorded pulses in chunks, correlate, and find matched pulses. Step 1 of a 2-step alignment process. 
    Step 2 uses these matched pulses for the regression and offset determination!
    
    """
    neural_blockstart = np.linspace(0, len(neural_ts)-window, window)
    beh_ipi = np.diff(beh_ts)
    neural_ipi = np.diff(neural_ts)

    print(f'{len(neural_blockstart)} blocks')
    blockR = [] 
    blockBehMatch = [] 

    for block in neural_blockstart:
        print('.', end =" ")
        neural_ix = np.arange(window-1) + block
        neural_d = neural_ipi[neural_ix.astype(int)]
        r = np.zeros(len(beh_ipi) - len(neural_d))
        p = r.copy() 
        for i in np.arange(len(beh_ipi)-len(neural_d)):
            temp_beh_ix = np.arange(window-1) + i 
            r_temp = np.corrcoef(neural_d, beh_ipi[temp_beh_ix])[0,1]
            r[i] = r_temp
        blockR.append(np.max(r))
        blockBehMatch.append(np.argmax(r))
    neural_offset = [] 
    good_beh_ms = [] 
    blockR = np.array(blockR)
    goodblocks = np.where(blockR>thresh)[0]
    for b in goodblocks:
        neural_ix = np.arange(window-1) + neural_blockstart[b]
        neural_offset.extend(neural_ts[neural_ix.astype(int)])
        beh_ix = np.arange(window-1) + blockBehMatch[b]
        good_beh_ms.extend(beh_ts[beh_ix])

    print(f'found matches for {len(goodblocks)} of {len(neural_blockstart)} blocks')

    return good_beh_ms, neural_offset

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