import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import mne


# There are some things that MNE is not that good at, or simply does not do. Let's write our own code for these. 

def lfp_sta(ev_times, signal, sr, pre, post):
    '''
    Compute the STA for a vector of stimuli.

    Input: 
    spikes - raw spike times used to compute STA, should be in s
    signal - signal for averaging. can be filtered or unfiltered.  
    bound - bound of the STA in ms, +- this number 
    
    '''

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

def plot_ERP(data, pre_win, post_win, sr, title):
    """

    """
    pass



def plot_TFR(data, freqs, pre_win, post_win, sr, title):
    """

    pre_win should be in seconds
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


def detect_ripple_evs(signal, method=None):
    
    """
    Input: 
    signal = continuous voltage time-series filtered and re-referenced
    
    Method 1: Foster et al., 
    1. band-pass filtered from 80 to 120 Hz (ripple band) using a 4th order FIR filter.
    2. the root mean square (RMS) of the band-passed signal was calculated and smoothed using a 20-ms window
    3. ripple events were identified as having an RMS amplitude above 2.5, but no greater than 9, standard deviations from the mean
    4. detected ripple events with a duration shorter than 38 ms (corresponding to 3 cycles at 80 Hz) or longer than 500 ms, were rejected.

In addition to the amplitude and duration criteria the spectral features of each detected ripple event were examined
    5. calculate the frequency spectrum for each detected ripple event by averaging the normalized instantaneous amplitude between the onset and offset of the ripple event for the frequency range of 2–200 Hz.
    6. spectral amplitude was normalized to a percent change signal by applying a baseline correction at each frequency based on the mean amplitude of the entire recording for a given electrode and frequency
    7. reject events with more than one peak in the ripple band
    8. reject events where the most prominent and highest peak was outside the ripple band and sharp wave band for a frequencies > 30 Hz
    9. reject events where ripple-peak width was greater than 3 standard deviations from the mean ripple-peak width calculated for a given electrode and recording session
    10. reject events where high frequency activity peaks exceed 80% of the ripple peak height
  
    """
    sf = signal.info["sfreq"] # get the sampling frequency
    single_ch = signal._data[1,:] # subset just one channel (dev purposes)
    
    # Step 1: band-pass filter from 80 - 120 Hz (ripple band) using a FIR filter
    bandpass_filtered_data = mne.filter.filter_data(single_ch,sf,80,120,method='fir')
    
    # visualize 2s of white matter referenced and band-passed signal
    fig1 = plt.figure()
    plt.plot(bandpass_filtered_data[1:2048]) # plot the first 2s of data
    plt.plot(bandpass_filtered_data[1:2048]) # plot the first 2s of data
    # plt.show() - for dev purposes  
    
    # Step 2: Calculate the root mean square of the band-passed signal and smooth using a 20 ms window
    column_values = ['signal']
    df = pd.DataFrame(data = bandpass_filtered_data, columns = column_values)
    smoothed_data = df['signal'].pow(2).rolling(round(0.02 * 1024),min_periods=1).mean().apply(np.sqrt, raw=True)

    # Step 3: mark ripple events [ripple start, ripple end] as periods of RMS amplitude above 2.5, but no greater than 9, standard deviations from the mean 

    # calculate mean and standard deviation of smoothed rms data
    smoothed_mean = np.mean(smoothed_data)
    smoothed_sd = np.std(smoothed_data)

    # calculate lower (above 2.5 SD from mean) and upper (lower than 9 SD from mean) cutoffs for marking ripple events 
    lower_cutoff = smoothed_mean + 2.5 * smoothed_sd
    upper_cutoff = smoothed_mean + 9 * smoothed_sd

    ripple_events_index = np.asarray(np.where((smoothed_data > lower_cutoff) & (smoothed_data < upper_cutoff)))

    # Step 4: detected ripple events with a duration shorter than 38 ms (corresponding to 3 cycles at 80 Hz) or longer than 500 ms, were rejected.
    min_length_ripple_event = 0.038 * sf
    max_length_ripple_event = 0.5 * sf
    
    ripple_events_differences = np.array([0] + np.diff(ripple_events_index[0,:]))

    # get the lengths and indices of consecutive 1s (this is how we know that they are sequential samples)
    _, idx, counts = np.unique(np.cumsum(1-ripple_events_differences)*ripple_events_differences, return_index=True, return_counts=True)    

    ripple_events_index_correct_time = idx[np.where((counts > min_length_ripple_event) & (counts < max_length_ripple_event))] # index of ripple events that reach criterion
    ripple_events_length_samples = counts[np.where((counts > min_length_ripple_event) & (counts < max_length_ripple_event))]  # length in samples of ripple events that reach criterion
    ripple_events_length_seconds = ripple_events_length/1024 # length in seconds of ripple events that reach criterion

    
    return ripple_rate, ripple_duration, ripple_peak_amp, ripple_peak_freq


def detect_oscillation_evs(signal, method=None): 
    """
    """
    pass 


def empirical_mode_decomposition(signal):
    """
    My preference is to eventually implement EMD because it is better for our stupid non-stationary data. 

    https://emd.readthedocs.io/en/stable/index.html
    """
    pass


def sliding_FOOOF(signal): 
    """
    Implement time-resolved FOOOF: 
    https://github.com/lucwilson/SPRiNT

    """
    pass


def hctsa_signal_features(signal): 
    """
    Implement https://github.com/DynamicsAndNeuralSystems/catch22
    """

