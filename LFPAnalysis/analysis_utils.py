import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import mne
import pickle 
import joblib
import fooof
from fooof import FOOOFGroup
from matplotlib.backends.backend_pdf import PdfPages
import os
import pycatch22

# There are some things that MNE is not that good at, or simply does not do. Let's write our own code for these. 
def select_rois_picks(elec_data, chan_name):
    """
    Grab specific roi for the channel you are looking at 
    """

    roi = np.nan
    NMM_label = elec_data[elec_data.label==chan_name].NMM.str.lower()
    if NMM_label.str.contains('entorhinal').iloc[0]:
            roi = 'entorhinal'
    else:
        YBA_label = elec_data[elec_data.label==chan_name].YBA_1.str.lower()
        if (YBA_label.str.contains('cingulate gyrus a').iloc[0]) | (YBA_label.str.contains('cingulate gyrus b').iloc[0]) | (YBA_label.str.contains('cingulate gyrus c').iloc[0]):
            roi = 'anterior_cingulate'
        elif (YBA_label.str.contains('hippocampus').iloc[0]):
            roi = 'hippocampus'
        elif (YBA_label.str.contains('amygdala').iloc[0]):
            roi = 'amygdala'
        elif (YBA_label.str.contains('insula').iloc[0]):
            roi = 'insula'
        elif (YBA_label.str.contains('parahippocampal').iloc[0]):
            roi = 'parahippocampal'

    return roi

def select_picks_rois(elec_data, roi=None):
    """
    Grab specific electrodes that you care about 
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
        elif entorhinal in roi: 
            roi.remove('entorhinal')
            picks_ec =  elec_data[elec_data.NMM.str.lower().str.contains('entorhinal')].label.tolist()
        picks = elec_df[elec_df.YBA_1.str.lower().str.contains('|'.join(roi))].label.tolist()
        if picks_ec is not None: 
            picks += picks_ec

    else:
        # Just grab everything 
        picks = elec_df.label.tolist()
    
    return picks 

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


def detect_ripple_evs(signal, min_ripple_length=0.038, max_ripple_length=0.5, smoothing_window_length=0.02, sd_upper_cutoff=9,sd_lower_cutoff=2.5,plotting_window = 0.20, rmethod=None):
    
    """
    Input: 
    signal = continuous voltage time-series filtered and re-referenced
    smoothing_window_length = window size in seconds for smoothing (relevant for step 2)
    sd_upper_cutoff = maximum standard deviations from the mean for RMS amplitude (relevant for step 3)
    sd_lower_cutoff = minimum standard deviations from the mean for RMS amplitude (relevant for step 3)
    min_ripple_length = min duration of ripple events in seconds (relevant for step 4)
    max_ripple_length = max duration of ripple events in seconds (relevant for step 4)
    plotting_window = window size (in seconds) for plotting average ripples
    
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
    signal = mne.io.read_raw_fif('/Users/christinamaher/Desktop/MS009/wm_ref_ieeg.fif', preload=True)
    sf = signal.info["sfreq"] # get the sampling frequency
    single_ch = signal._data[1,:] # subset just one channel (dev purposes)
    
    # Step 1: band-pass filter from 80 - 120 Hz (ripple band) using a FIR filter
    bandpass_filtered_data = mne.filter.filter_data(single_ch,sf,80,120,method='fir')
    
    # visualize 2s of white matter referenced and band-passed signal
    fig1 = plt.figure()
    plt.plot(single_ch[1:2048]) # plot the first 2s of wm re-referenced data 
    plt.plot(bandpass_filtered_data[1:2048]) # plot the first 2s of bandpass filted + wm re-referenced data
    plt.show() #- for dev purposes  
    
    # Step 2: Calculate the root mean square of the band-passed signal and smooth using a 20 ms window
    column_values = ['signal'] 
    df = pd.DataFrame(data = bandpass_filtered_data, columns = column_values)
    smoothed_data = df['signal'].pow(2).rolling(round(smoothing_window_length * sf),min_periods=1).mean().apply(np.sqrt, raw=True)

    # Step 3: mark ripple events [ripple start, ripple end] as periods of RMS amplitude above 2.5, but no greater than 9, standard deviations from the mean 

    # calculate mean and standard deviation of smoothed rms data
    smoothed_mean = np.mean(smoothed_data)
    smoothed_sd = np.std(smoothed_data)

    # calculate lower (above 2.5 SD from mean) and upper (lower than 9 SD from mean) cutoffs for marking ripple events 
    lower_cutoff = smoothed_mean + sd_lower_cutoff * smoothed_sd
    upper_cutoff = smoothed_mean + sd_upper_cutoff * smoothed_sd

    ripple_events_index = np.asarray(np.where((smoothed_data > lower_cutoff) & (smoothed_data < upper_cutoff)))

    # Step 4: detected ripple events with a duration shorter than 38 ms (corresponding to 3 cycles at 80 Hz) or longer than 500 ms, were rejected.
    min_length_ripple_event = min_ripple_length * sf # add an input parameter for duration of ripple event
    max_length_ripple_event = max_ripple_length * sf
    
    ripple_events_differences = np.array([0] + np.diff(ripple_events_index[0,:]))

    # get the lengths and indices of consecutive 1s (this is how we know that they are sequential samples)
    _, idx, counts = np.unique(np.cumsum(1-ripple_events_differences)*ripple_events_differences, return_index=True, return_counts=True)    

    ripple_events_index_correct_time = idx[np.where((counts > min_length_ripple_event) & (counts < max_length_ripple_event))] # index of ripple events that reach criterion
    ripple_events_length_samples = counts[np.where((counts > min_length_ripple_event) & (counts < max_length_ripple_event))]  # length in samples of ripple events that reach criterion
    ripple_end_index = ripple_events_index_correct_time + ripple_events_length_samples
    ripple_events_length_seconds = ripple_events_length_samples/1024 # length in seconds of ripple events that reach criterion
    
    # zip the three lists using zip() function --> ripple_results is a list of tuples containing the starting index of each ripple, the ending index of each ripple, and the length of each ripple in seconds
    ripple_results = list(zip(ripple_events_index_correct_time,ripple_end_index,ripple_events_length_seconds))
    num_ripples = len(ripple_results) # this is the number of ripples
    
    # plot each ripple individually
    i = 0
    while i < num_ripples:
        ripple_temp = ripple_results[i]
        plt.figure()
        plt.plot(smoothed_data[ripple_temp[0]:ripple_temp[1]])
        i += 1
    plt.show() # calling this outside the loop shows all plots at once 

    # plot the average of ripples using bandpass filtered data (pre-smoothing window)
    #ripple_matrix = np.zeros((num_ripples, ((round(plotting_window * sf))*2)), dtype=np.int32) # nrows = number of ripples, ncols = maximum number of samples in a given ripple for this electrode
    longest_ripple_index = np.array(np.where(ripple_events_length_samples == max(ripple_events_length_samples))).item() # appending .item() returns as a scalar which can be used more flexibly for indexing etc.
    longest_ripple_info = ripple_results[longest_ripple_index]
    longest_ripple_data = smoothed_data[longest_ripple_info[0]:longest_ripple_info[1]]
    longest_ripple_peak_index = np.array(np.where(longest_ripple_data == max(longest_ripple_data))).item()


    i = 0 
    ripple_matrix = []
    while i < num_ripples:
        ripple_info = ripple_results[i]
        ripple_temp = bandpass_filtered_data[ripple_info[0]:ripple_info[1]]
        ripple_peak_index = np.array(np.where(bandpass_filtered_data == max(ripple_temp))).item()
        num_samples_plotting = round(plotting_window * sf)
        min_sample_index = ripple_peak_index - num_samples_plotting
        max_sample_index = ripple_peak_index + num_samples_plotting
        ripple_temp = np.array(bandpass_filtered_data[min_sample_index:max_sample_index])
        ripple_matrix.append(ripple_temp)
        i += 1

    avg_ripple_by_elec = np.mean(ripple_matrix, axis = 0) 
    
    plt.figure()
    plt.plot(avg_ripple_by_elec)
    plt.show()

    # function for plotting ripples (sanity check) - find the longest ripple and make that the matrix size, ripples that are shorter will be padded by NAs so they are still centered. 

    return ripple_rate, ripple_duration, ripple_peak_amp, ripple_peak_freq

def FOOOF_continuous(signal):
    """
    TODO
    """
    pass 


def FOOOF_compute_epochs(epochs, tmin=0, tmax=1.5, **kwargs):
    """

    This function is meant to enable easy computation of FOOOF. 

    Parameters
    ----------
    epochs : mne Epochs object 
        mne object

    tmin : time to start (s) 
        float

    tmax : time to end (s) 
        float

    band_dict : definitions of the bands of interest
        dict

    kwargs : input arguments to the FOOOFGroup function, including: 'min_peak_height', 'peak_threshold', 'max_n_peaks'
        dict 

    Returns
    -------
    mne_data_reref : mne object 
        mne object with re-referenced data
    """

    # bands = fooof.bands.Bands(band_dict)

    epo_spectrum = epochs.compute_psd(method='multitaper',
                                                tmin=tmin,
                                                tmax=tmax)
                                                
    psds, freqs = epo_spectrum.get_data(return_freqs=True)
            
    # average across epochs
    psd_trial_avg = np.average(psds, axis=0) 

    # Initialize a FOOOFGroup object, with desired settings
    FOOOFGroup_res = FOOOFGroup(peak_width_limits=kwargs['peak_width_limits'], 
                    min_peak_height=kwargs['min_peak_height'],
                    peak_threshold=kwargs['peak_threshold'], 
                    max_n_peaks=kwargs['max_n_peaks'], 
                    verbose=False)

    # Fit the FOOOF object 
    FOOOFGroup_res.fit(freqs, psd_trial_avg, kwargs['freq_range'])

    all_chan_dfs = []
    # Go through individual channels
    for chan in range(len(epochs.ch_names)):

        ind_fits = FOOOFGroup_res.get_fooof(ind=chan, regenerate=True)
        ind_fits.fit()

        # Create a dataframe to store results 
        chan_data_df = pd.DataFrame(columns=['channel', 'frequency', 'PSD_raw', 'PSD_corrected', 'in_FOOOF_peak', 'peak_freq', 'peak_height', 'PSD_exp'])

        chan_data_df['frequency'] = ind_fits.freqs.tolist()
        chan_data_df['PSD_raw'] = ind_fits.power_spectrum.tolist()
        chan_data_df['PSD_corrected'] = ind_fits._spectrum_flat.tolist()

        # Get aperiodic exponent
        exp = ind_fits.get_params('aperiodic_params', 'exponent')
        chan_data_df['PSD_exp'] = exp

        # Get peak info
        peaks = fooof.analysis.get_band_peak_fm(ind_fits, band=(1, 30), select_highest=False)
        in_FOOOF_peaks = [] 
        peak_freqs = []
        peak_heights = []
        # in_FOOOF_peak = np.zeros_like(ind_fits.freqs)
        # peak_freq = np.ones_like(ind_fits.freqs) * np.nan
        # peak_height = np.ones_like(ind_fits.freqs) * np.nan
        
        # Iterate through the peaks and create dataframe friendly data that assigns each frequency to a peak (or not)
        if np.ndim(peaks) == 1: # only one peak

            center_pk = peaks[0]
            low_freq = peaks[0] - (peaks[2]/2)
            high_freq = peaks[0] + (peaks[2]/2)
            pk_height = peaks[1]
            in_FOOOF_peak = np.zeros_like(ind_fits.freqs)
            in_FOOOF_peak[(ind_fits.freqs>=low_freq) & (ind_fits.freqs<=high_freq)] = 1
            peak_freq = np.zeros_like(ind_fits.freqs)
            peak_freq[(ind_fits.freqs>=low_freq) & (ind_fits.freqs<=high_freq)] = center_pk
            peak_height = np.zeros_like(ind_fits.freqs)
            peak_height[(ind_fits.freqs>=low_freq) & (ind_fits.freqs<=high_freq)] = pk_height

            in_FOOOF_peaks.append(in_FOOOF_peak)
            peak_freqs.append(peak_freq)
            peak_heights.append(peak_height)   

        elif np.ndim(peaks) > 1: # more than one peak
            for ix, pk in enumerate(peaks):
                center_pk = pk[0]
                low_freq = pk[0] - (pk[2]/2)
                high_freq = pk[0] + (pk[2]/2)
                pk_height = pk[1]
                in_FOOOF_peak = np.zeros_like(ind_fits.freqs)
                in_FOOOF_peak[(ind_fits.freqs>=low_freq) & (ind_fits.freqs<=high_freq)] = ix + 1
                peak_freq = np.zeros_like(ind_fits.freqs)
                peak_freq[(ind_fits.freqs>=low_freq) & (ind_fits.freqs<=high_freq)] = center_pk
                peak_height = np.zeros_like(ind_fits.freqs)
                peak_height[(ind_fits.freqs>=low_freq) & (ind_fits.freqs<=high_freq)] = pk_height

                in_FOOOF_peaks.append(in_FOOOF_peak)
                peak_freqs.append(peak_freq)
                peak_heights.append(peak_height)

        if peaks is not None:
            in_FOOOF_peaks = sum(in_FOOOF_peaks)
            peak_freqs = sum(peak_freqs)
            peak_heights = sum(peak_heights)

        chan_data_df['in_FOOOF_peak'] = in_FOOOF_peaks
        chan_data_df['peak_freq'] = peak_freqs
        chan_data_df['peak_height'] = peak_heights

        # , 'peak_pow', 'band_pow', 'band_pow_flat', 'band', 'exp'

        # 'frequency', 'PSD_raw', 'PSD_corrected', 'in_FOOOF_peak', 'PSD_exp'  

        # band_labels = []
        # peak_pow = [] 
        # band_pow = []
        # band_pow_flats = []

        # for label, definition in bands:
        #     band_labels.append(label)
        #     peak_pow.append(fooof.analysis.get_band_peak_fm(ind_fits, definition)[1])
        #     band_pow.append(np.mean(fooof.utils.trim_spectrum(ind_fits.freqs, ind_fits.power_spectrum, definition)[1]))
        #     band_pow_flats.append(np.mean(fooof.utils.trim_spectrum(ind_fits.freqs, ind_fits._spectrum_flat, definition)[1]))

        # chan_data_df['peak_pow'] = peak_pow
        # chan_data_df['band_pow'] = band_pow
        # chan_data_df['band_pow_flat'] = band_pow_flats
        # chan_data_df['band'] = band_labels
        # chan_data_df['exp'] = exp
        chan_data_df['channel'] = epochs.ch_names[chan]
        # chan_data_df['region'] = epochs.metadata.region.unique()[0]

        all_chan_dfs.append(chan_data_df)


    return FOOOFGroup_res, pd.concat(all_chan_dfs)


# def FOOOF_compare_epochs(epochs_with_metadata, tmin=0, tmax=1.5, conditions=None, band_dict=None, 
# file_path=None, plot=True, **kwargs):
#     """
#     Function for comparing conditions.
#     """
#     # Helper functions for computing and analyzing differences between power spectra. 

#     t_settings = {'fontsize' : 24, 'fontweight' : 'bold'}
#     # If the path doesn't exist, make it:
#     if not os.path.exists(file_path): 
#         os.makedirs(file_path)

#     # def _compare_exp(fm1, fm2):
#     #     """Compare exponent values."""

#     #     exp1 = fm1.get_params('aperiodic_params', 'exponent')
#     #     exp2 = fm2.get_params('aperiodic_params', 'exponent')

#     #     return exp1 - exp2

#     # def _compare_peak_pw(fm1, fm2, band_def):
#     #     """Compare the power of detected peaks."""

#     #     pw1 = fooof.analysis.get_band_peak_fm(fm1, band_def)[1]
#     #     pw2 = fooof.analysis.get_band_peak_fm(fm2, band_def)[1]

#     #     return pw1 - pw2

#     # def _compare_band_pw(fm1, fm2, band_def):
#     #     """Compare the power of frequency band ranges."""

#     #     pw1 = np.mean(fooof.utils.trim_spectrum(fm1.freqs, fm1.power_spectrum, band_def)[1])
#     #     pw2 = np.mean(fooof.utils.trim_spectrum(fm1.freqs, fm2.power_spectrum, band_def)[1])

#     #     return pw1 - pw2

#     # def _compare_band_pw_flat(fm1, fm2, band_def):
#     #     """Compare the power of frequency band ranges."""

#     #     pw1 = np.mean(fooof.utils.trim_spectrum(fm1.freqs, fm1._spectrum_flat, band_def)[1])
#     #     pw2 = np.mean(fooof.utils.trim_spectrum(fm1.freqs, fm2._spectrum_flat, band_def)[1])

#     #     return pw1 - pw2

#     # shade_cols = ['#e8dc35', '#46b870', '#1882d9', '#a218d9', '#e60026']
#     # bands = fooof.bands.Bands(band_dict)
#     all_chan_dfs = [] 

#     fooof_groups_cond = {f'{x}': np.nan for x in conditions}

#     all_cond_df = []
#     for cond in conditions: 
#         # check that this is an appropriate parsing (is it in the metadata?)
#         try:
#             epochs_with_metadata.metadata.query(cond)
#         except pd.errors.UndefinedVariableError:
#             raise KeyError(f'FAILED: the {cond} condition is missing from epoch.metadata')
    
#         FOOOFGroup_res, cond_df = FOOOF_compute_epochs(epochs_with_metadata[cond], tmin=0, tmax=1.5, **kwargs)

#         fooof_groups_cond[cond] = FOOOFGroup_res


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

def detect_oscillation_evs(signal, method=None): 
    """
    Eventually could port: https://github.com/neurofractal/fBOSC
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
    https://github.com/lucwilson/SPRiNT now has a python implementation we can borrow from! 
    

    """
    pass


def hctsa_signal_features(signal): 
    """
    Implement https://github.com/DynamicsAndNeuralSystems/catch22
    """

    signal_features = pycatch22.catch22_all(signal)

    # Data organization
    df = pd.DataFrame.from_dict(signal_features, orient='index')
    df.columns = df.iloc[0]
    df.reset_index(inplace=True, drop=True)
    df = df.drop(labels=0, axis=0).reset_index(drop=True)

    return df 
