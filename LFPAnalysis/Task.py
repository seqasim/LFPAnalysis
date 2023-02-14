# Task class 

class Task(object):
    """
    This is a base class for experiments. The idea is to get and set certain properties, and re-use task-specific methods. 
    """

    def __init__(self) -> None:
        pass

    @property
    def behavioral_data(self): 
        print('Setting behavioral data')
        return self._behavioral_data

    @behavioral_data.setter
    def behavioral_data(self):
        self._behavioral_data = None

    @property
    def neural_data(self): 
        print('Setting neural data')
        return self._neural_data

    @neural_data.setter
    def neural_data(self):
        self._neural_data = None

    @property
    def behavioral_timestamps(self): 
        print('Setting behavioral timestamps')
        return self._behavioral_timestamps

    @behavioral_timestamps.setter
    def behavioral_timestamps(self):
        self._behavioral_timestamps = None


    @property
    def neural_timestamps(self): 
        print('Setting neural timestamps')
        return self._neural_timestamps

    @neural_timestamps.setter
    def neural_timestamps(self):
        self._neural_timestamps = None


    @staticmethod
    def moving_average(a, n=11) :
        """
        Clean up the sync channel a bit. 
        """
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def pulsealign(self, window=30, thresh=0.99):
    """
    Step through recorded pulses in chunks, correlate, and find matched pulses. Step 1 of a 2-step alignment process. 
    Step 2 uses these matched pulses for the regression and offset determination!
    
    """
        neural_blockstart = np.linspace(0, len(self.neural_ts)-window, window)
        beh_ipi = np.diff(self.beh_ts)
        neural_ipi = np.diff(self.neural_ts)

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
            neural_offset.extend(self.neural_ts[neural_ix.astype(int)])
            beh_ix = np.arange(window-1) + blockBehMatch[b]
            good_beh_ms.extend(self.beh_ts[beh_ix])

        print(f'found matches for {len(goodblocks)} of {len(neural_blockstart)} blocks')

        return good_behavioral_syncs, good_neural_syncs


    @property
    def sync_info(self): 
        print('Setting synchronization info')
        return self._sync_info

    @sync_info.setter
    def sync_info(self):
        """
        This function should set the slope and offset computation needed to synchronize the behavioral and neural data. 
        """
        good_behavioral_syncs, good_neural_syncs = pulsealign()
        
        bfix = good_behavioral_syncs[0]
        res = scipy.stats.linregress(good_behavioral_syncs-bfix, good_neural_syncs)
        slope = res[0]
        offset = res[1]
        offset = offset - bfix*slope
        rval = res[2]

        self._sync_info = {'slope': slope,
        'offset':offset,
        'rval':rval}
    

        
