import mne 
import pandas as pd
from glob import glob
import scipy
import os 

class Subject(object):
    """
    Each person should be 
    """
    def __init__(self, subj_id=None, task_id=None, base_path='/sc/arion/', reref='wm'):

        self.subj_id = subj_id 
        self.task_id = task_id 

        # Set all the paths. Override this with your own paths: 
        self.base_path = base_path
        # All the relevant directories 
        self.paths = {'neural_paths':[f'{self.base_path}/projects/guLab/Salman/EMU/{subj_id}/neural/Day1', 
        f'{self.base_path}/projects/guLab/Salman/EMU/{subj_id}/neural/Day2'],
        'behav_paths':[f'{self.base_path}/projects/guLab/Salman/EMU/{subj_id}/behav/Day1', 
        f'{self.base_path}/projects/guLab/Salman/EMU/{subj_id}/behav/Day2'],
        'elec_path':f'{self.base_path}/projects/guLab/Salman/EMU/{subj_id}/anat/',
        'save_path':f'{self.base_path}/work/qasims01/MemoryBanditData/EMU/Subjects/{subj_id}/'}

        for filepaths in self.paths.values():
            # make directories if missing
            if not os.path.exists(os.path.split(filepaths)[0]):
                try:
                    os.makedirs(os.path.split(filepaths)[0])
                except OSError:
                    pass
            if not os.path.exists(filepaths):
                try:
                    os.makedirs(filepaths)
                except OSError:
                    pass

        # Search for .csv file 
        elec_files = glob(f'{self.paths['elec_path']}/*.csv')[0]
        elec_data = pd.read_csv(self.paths['elec_path'])
        # Sometimes there's extra columns with no entries: 
        self.elec_data = elec_data[elec_data.columns.drop(list(elec_data.filter(regex='Unnamed')))]

        self.reref = reref
        self.mne_reref = None

    def reref_mne(self, format='edf', site='MSSM', overwrite=True):
        
        for filepath in 
        # The following code also splits out the photodiode
        mne_raw = lfp_preprocess_utils.make_mne(load_path=self.paths['neural_paths'], 
        elec_data=self.elec_data, 
        format=format)

        mne_reref = lfp_preprocess_utils.ref_mne(mne_data=self.mne_raw, 
                                                elec_data=self.elec_data, 
                                                method=self.reref, 
                                                site=site)

        mne_data_reref.save(f'{self.paths['save_paths']}/{self.reref}_ref_ieeg.fif', 
        overwrite=overwrite)

    def 


    @property
    def photodiode(self): 
        print('Setting photodiode')
        return self._photodiode

    @photodiode.setter
    def photodiode(self):
        self._photodiode = mne.io.read_raw_fif(f'{self.load_path}/photodiode.fif', preload=True)

    @property
    def mne_reref(self): 
        print('Setting mne_reref')
        return self._mne_reref

    @mne_reref.setter
    def mne_reref(self):
        self._mne_reref = mne.io.read_raw_fif(f'{self.save_path}/{self.reref}_ref_ieeg.fif', preload=True)

    @property
    def neural_ts(self): 
        print('Setting neural_ts')
        return self._neural_ts

    @neural_ts.setter
    def neural_ts(self):
        sig = np.squeeze(sync_utils.moving_average(self._photodiode._data, n=11))
        timestamp = np.squeeze(np.arange(len(sig))/self._photodiode.info['sfreq'])
        # normalize
        sig =  scipy.stats.zscore(sig)
        # look for z-scores above 1
        trig_ix = np.where((sig[:-1]<=0)*(sig[1:]>0))[0] # rising edge of trigger
        neural_ts = timestamp[trig_ix]
        self._neural_ts = np.array(neural_ts)
        return self._neural_ts