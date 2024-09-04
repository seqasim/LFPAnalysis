# In this libray I want to write some functions that make certain statistical analyses easier to run 

import scipy as sp
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool

import patsy
from statsmodels.api import OLS
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_1samp
from scipy.ndimage import label

import warnings 
warnings.filterwarnings('ignore')

def fit_permuted_model(y_permuted, X):
    """
    Convenience function for running backend OLS with surrogates
    """
    return OLS(y_permuted, X).fit().params

def permutation_regression_zscore(data, formula, n_permutations=1000, plot_res=False):
    """

    A quick way to perform single-electrode regression with many permutations: 
    # Example usage:
    # data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'category': ['A', 'B', 'A', 'B', ...]})
    # formula = 'y ~ x1 + x2 + C(category)'
    # results = permutation_regression_zscore(data, formula, plot_res=True)
    # print(results)

    """
    # Perform original regression
    y, X = patsy.dmatrices(formula, data, return_type='dataframe')
    original_model = OLS(y, X).fit()
    
    # Extract original coefficients
    original_params = original_model.params
    
    # Prepare data for permutations
    y_values = y.values.ravel()
    X_values = X.values
    
    # Perform permutations
    permuted_params = []
    permuted_y_values = []
    for _ in tqdm(range(n_permutations), desc="Permutations"):
        y_permuted = np.random.permutation(y_values)
        permuted_params.append(fit_permuted_model(y_permuted, X_values))
        permuted_y_values.append(y_permuted)
    
    # Convert to numpy array for faster computations
    permuted_params = np.array(permuted_params)
    
    # Compute z-scores
    permuted_means = np.mean(permuted_params, axis=0)
    permuted_stds = np.std(permuted_params, axis=0)
    z_scores = (original_params - permuted_means) / permuted_stds
    
    # Compute p-values from z-scores
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    # Prepare results
    results = pd.DataFrame({
        'Original_Estimate': original_params,
        'Permuted_Mean': permuted_means,
        'Permuted_Std': permuted_stds,
        'Z_Score': z_scores,
        'P_Value': p_values
    })
    
    # Plotting
    if plot_res:
        features = [col for col in X.columns if col != 'Intercept']
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(3*n_features, 3*n_features), squeeze=False, dpi=300)
        
        for i, feature in enumerate(features):
            ax = axes[i, 0]
            
            # Plot permuted data first (in black)
            for j in range(min(100, n_permutations)):  # Limit to 100 permutations for clarity
                sns.regplot(x=X[feature], y=permuted_y_values[j], ax=ax, scatter=False,
                            line_kws={'color': 'black', 'alpha': 0.05}, ci=None)
            
            # Plot original data (in red)
            sns.regplot(x=X[feature], y=y_values, ax=ax, scatter_kws={'alpha': 0.5}, 
                        line_kws={'color': 'red', 'label': 'Original'}, ci=None)
            
            # Add z-score and p-value to the plot
            orig_param = original_params[i+1]
            z_score = z_scores[i+1]
            p_value = p_values[i+1]
            ax.text(0.05, 0.95, f'Beta: {orig_param:.2f}\nZ-score: {z_score:.2f}\np-value: {p_value:.3f}', 
                    transform=ax.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{feature} vs y')
            ax.legend()
            
            # Despine the plot
            sns.despine(ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    return results

def time_resolved_regression_single_channel(timeseries=None, regressors=None, 
                             win_len=100, slide_len=25, ts_index=None,
                             standardize=True, smooth=False, permute=False, sr=500):
    """
    In this function, if you provide a 2D array of z-scored time-varying neural data and a sert of regressors, 
    this function will run a time-resolved generalized linear model with the provided regressor dataframe. 

    Typically, this timeseries will be HFA, and the default win_len and slide_len reflect this 

    timeseries: ndarray, trials x times 
    regressors: pandas df, index = trials, columns = regressors

    Parameters
    ----------
    timeseries : 2D ndarray, dimensions = trials x times
        Time-varying neural data.
    regressors : pandas.DataFrame, dimensions = trials x regressors
        Dataframe containing the regressors.
    win_len : int
        Length of the window for the time-resolved regression.
    slide_len : int
        Step size for the time-resolved regression.
    standardize : bool
        Whether to standardize the regressors. The default is True.
    standardize : bool
        Whether to bin the timeseries according to win_len and slide_len. The default is Fault.
    sr: int
        sampling rate to determine the proper timing of the resulting timeseries of coefficients
    """

    # Optional: standardize the regressors
    if standardize:
        regressors = regressors.apply(lambda x: (x - np.nanmean(x))/(2*np.nanstd(x)))

    # Optional: bin the data
    if smooth: 
        slices = np.lib.stride_tricks.sliding_window_view(np.arange(timeseries.shape[-1]), win_len)[::slide_len]
        midpoints = np.ceil(np.mean(slices, axis=1))
        # Smooth the timeseries (easier to do here than to store smoothed data)
        sig = np.zeros([timeseries.shape[0], slices.shape[0]])
        for stride in range(slices.shape[0]):
            sig[:, stride] = np.nanmean(timeseries[:, slices[stride]], axis=1)
            # [np.nanmean(timeseries[trial, i:i+win_len]) for i in range(0, timeseries.shape[1], slide_len) if i+win_len <= timeseries.shape[1]]
    else:
        sig = timeseries

    all_res = []
    # write the regression formula
    formula = f'sig ~ 1+'+'+'.join(regressors.keys())
    for ts in range(sig.shape[-1]):
        regressors['sig'] = sig[:, ts]
        if permute:
            results = permutation_regression_zscore(regressors,
            formula,
            n_permutations=500, 
            plot_res=False)
            results['ts'] = ts
        else:
            y, X = patsy.dmatrices(formula, regressors, return_type='dataframe')
            original_model = OLS(y, X).fit()
            # Prepare results
            results = pd.DataFrame({
                'Original_Estimate': original_model.params,
                'Original_BSE': original_model.bse,
                'P_Value': original_model.pvalues,
                'ts': ts 
            })
        all_res.append(results)
        
    all_res = pd.concat(all_res)

    if smooth: # assign the timestamp to the middle of each bin in samples
        for ts_i in all_res.ts.unique():
            all_res.loc[all_res.ts==ts_i, 'ts'] = midpoints[ts_i] * (1000/sr)
    else:
        all_res['ts'] = all_res['ts'] * (1000/sr)

    if ts_index is not None:
        all_res['time'] = [ts_index[int(t*(sr/1000))] for t in ts['ts'].values]
    
    return all_res

def find_clusters(roi_df, t_col='ts', cluster_p_thr=0.05, min_cluster_size=3):
    """
    Computes the one-sample t-test for each unique ts in the ROI DataFrame,
    identifies clusters of significant ts values, and sums the t-statistics
    within each cluster.
    
    Args:
        roi_df (pd.DataFrame): DataFrame with 'Original_Estimate' and 'ts' columns.
        t_col (str): Name of the column containing the time samples (default: 'ts')
        cluster_p_thr (float): Threshold for statistical significance (default: 0.05)
        min_cluster_size (int): Minimum number of consecutive significant ts values (e.g., `3`)

    Returns:
        pd.DataFrame: DataFrame with summed t-statistics for each cluster.
    """
    # sort df by t_col
    roi_df = roi_df.sort_values(t_col)
    
    # One-sample t-test at each ts
    roi_ttest = roi_df.groupby(t_col).apply(
        lambda x: pd.Series(
            {
                'tstat': ttest_1samp(x['Original_Estimate'], 0)[0],
                'pval': ttest_1samp(x['Original_Estimate'], 0)[1]
            }
        )
    ).reset_index()
    
    # Identify significant clusters (p < p_thr)
    roi_ttest['mask'] = roi_ttest['pval'] < cluster_p_thr

    # Identify and label non-overlapping clusters of 1s
    clusters, num_clusters = label(roi_ttest['mask'])

    # Extract cluster indices and labels
    cluster_info = [(i+1, np.where(clusters == i+1)[0].tolist()) for i in range(num_clusters)]

    # Initialize the 'cluster' column with NaN values
    roi_ttest['cluster'] = np.nan

    # Assign cluster labels to the corresponding indices
    for cluster_id, indices in cluster_info:
        roi_ttest.loc[indices, 'cluster'] = cluster_id
    
    # Sum t-stats for each cluster and pick the largest cluster
    cluster_tstats = roi_ttest.groupby('cluster').apply(
        lambda x: pd.Series(
            {
                'sum_tstat': x['tstat'].sum(),
                'abs_sum_tstat': np.abs(x['tstat']).sum(),
                'start_ts': x[t_col].min(),
                'end_ts': x[t_col].max(),
                'cluster_size': len(x)
            }
        )
    )

    # Filter clusters by minimum size
    if min_cluster_size:
        cluster_tstats = cluster_tstats[cluster_tstats.cluster_size >= min_cluster_size]
    
    return roi_ttest, cluster_tstats

def ts_permutation_test(roi_df, n_permutations=1000, t_col='ts', cluster_p_thr=.05, min_cluster_size=3, n_jobs=-1):
    """
    Performs permutation testing by shuffling the 'ts' variable and generating a null
    distribution of the summed t-statistics of the largest cluster using parallel processing.
    
    Args:
        roi_df (pd.DataFrame): DataFrame with 'Original_Estimate' and 'ts' columns.
        n_permutations (int): Number of permutations to perform.
        t_col (str): Name of the column containing the time samples (default: 'ts').
        cluster_p_thr (float): Threshold for statistical significance (default: 0.05).
        min_cluster_size (int): Minimum number of consecutive significant ts values (e.g., `3`).
        n_jobs (int): Number of parallel jobs (cores) to use. Default is -1 (use all available cores).
        
    Returns:
        list: Null distribution of the summed t-statistics of the largest cluster.
    """
    def get_largest_cluster_stat(roi_df, t_col='ts', cluster_p_thr=0.05, min_cluster_size=3, random_state=None):
        """
        Performs a single permutation test by shuffling the 'ts' values and computing the largest summed t-statistic.
        
        Args:
            roi_df (pd.DataFrame): DataFrame with 'Original_Estimate' and 'ts' columns.
            t_col (str): Name of the column containing the time samples (default: 'ts').
            cluster_p_thr (float): Threshold for statistical significance (default: 0.05).
            min_cluster_size (int): Minimum number of consecutive significant ts values (e.g., `3`).
            random_state (int): Random seed for reproducibility.
        
        Returns:
            float: The summed t-statistic of the largest cluster from the permuted data.
        """
        if random_state:
            np.random.seed(random_state)

        shuffled_df = roi_df.copy()
        shuffled_df[t_col] = np.random.permutation(shuffled_df[t_col])
        
        # Compute cluster stats for the shuffled data
        _, permuted_clusters = find_clusters(shuffled_df, t_col=t_col, cluster_p_thr=cluster_p_thr, min_cluster_size=min_cluster_size)
        
        # Find the largest summed t-stat cluster in permuted data
        if not permuted_clusters.empty:
            largest_cluster_sum_tstat = permuted_clusters['abs_sum_tstat'].max()
        else:
            largest_cluster_sum_tstat = 0  # Handle case with no clusters
        
        return largest_cluster_sum_tstat

    # Parallel computation of permutation tests
    null_distribution = Parallel(n_jobs=n_jobs)(
        delayed(get_largest_cluster_stat)(roi_df, t_col=t_col, cluster_p_thr=cluster_p_thr, min_cluster_size=min_cluster_size, random_state=iter) for iter in tqdm(range(n_permutations))
    )

    return null_distribution

def cluster_based_permutation(roi_df, n_permutations=1000, t_col='ts', cluster_p_thr, fwe_thr, min_cluster_size=3, output_null=False):
    """
    Perform cluster-based permutation test across all electrodes at each timepoint and compute FWE p-values.

    Parameters:
    - roi_df: DataFrame containing the region of interest (ROI) data.
    - n_permutations: Number of permutations for the null distribution (default: 1000).
    - t_col: Name of the column containing time data (default: 'ts').
    - cluster_p_thr: cluster forming threshold.
    - fwe_thr: Family-wise error rate threshold for significance.
    - min_cluster_size: Minimum number of consecutive significant ts values (default: 3).
    - output_null: Whether to output the null distribution (default: False).

    Returns:
    - roi_ttest: DataFrame with t-statistics and p-values for each timepoint.
    - cluster_tstats: DataFrame with summed t-statistics for each cluster identified.
    - null_distribution: List of summed t-statistics


    Example Usage:
        # can use outputs from `statistics_utils.time_resolved_regression_single_channel(timeseries,regressors,standardize=True,smooth=False)`
        ```
        roi_df = all_channels_df[all_channels_df.region == 'AMY'][['Original_Estimate', 'ts']]
        roi_ttest, cluster_tstats = cluster_based_permutation(roi_df)
        ```
    """

    # Compute actual statistics
    roi_ttest, cluster_tstats = find_clusters(roi_df, t_col=t_col, cluster_p_thr=cluster_p_thr, min_cluster_size=min_cluster_size)

    # Generate permutation-based null distribution
    null_distribution = ts_permutation_test(roi_df, n_permutations=n_permutations, t_col=t_col, p_thr=cluster_p_thr, min_cluster_size=min_cluster_size)

    # Add min and max values of the null distribution to cluster statistics
    cluster_tstats['null_min'] = np.min(null_distribution)
    cluster_tstats['null_max'] = np.max(null_distribution)

    # Compute FWE-corrected p-values and mask based on significance threshold
    cluster_tstats['fwe_pval'] = cluster_tstats['abs_sum_tstat'].apply(
        lambda x: np.mean(np.abs(null_distribution) >= np.abs(x))
    )
    cluster_tstats['fwe_mask'] = cluster_tstats['fwe_pval'] < fwe_thr

    if output_null:
        return roi_ttest, cluster_tstats, null_distribution
    else:
        return roi_ttest, cluster_tstats

# def time_resolved_regression_perm(timeseries=None, regressors=None, win_len=100, slide_len=25, standardize=True, sr=None, nsurr=500):

#     """
#     This utilizes the prior function to run permutations.

#     Parameters
#     ----------
#     nsurr : int 
#         Number of permutations to run. 
#     """

#     regress_df = time_resolved_regression(timeseries, regressors, win_len, slide_len, standardize, sr)

#     # Generate permuted timeseries 
#     shuffles = np.random.randint(1, timeseries.shape[1], nsurr)
#     all_surrs = []

#     # the following is a bit hacky if running in parallel        
#     # progress_bar = tqdm(np.arange(nsurr), ascii=True, desc='Computing Surrogate Regressions', position=0, leave=True)

#     # This one line is important. We shuffle in time, AND across trials! 
#     surrogate_data_list = [np.random.permutation(np.roll(timeseries, shuffles[surr], axis=1)) for surr in range(nsurr)]
    
#     for surr in range(nsurr): 
#         # Re-run regression with permuted timeseries 
#         surr_df = time_resolved_regression(surrogate_data_list[surr], regressors, win_len, slide_len, standardize, sr)
#         surr_df['nsurr'] = surr
#         all_surrs.append(surr_df)

#     all_surrs = pd.concat(all_surrs)

#     return all_surrs

# # Now we want to put this all together into a slightly clunky function that is meant to be used for running the regression
# # over multiple channels in parallel using joblib/Dask/multiprocessing.Pool: 
    
# def compute_time_resolved_regression_parallel(chan_name, TFR_object, subj_id, elec_df, event_name, bands=['hfa'],
#                              win_len=100, step_size=25, nsurr=500, save_path='/sc/arion/projects/guLab/Salman/EphysAnalyses',
#                              do_save=False):
#     """

#     Turn the TFR object into a dataframe, extract the time-resolved features, 
#     and run a time-resolved regression at whatever bands you want. 

#     Note: If using more than one band, we will need more multiple comparisons correction.

#     Parameters
#     ----------
#     chan_name : str
#         Name of the channel to be analyzed.
#     TFR_object : mne.time_frequency.AverageTFR
#         TFR object to be analyzed.
#     subj_id : str
#         Subject ID.
#     elec_df : pandas.DataFrame
#         Dataframe containing electrode information.
#         Note: input this way (rather than a region str) to enable easy parallelism 
#     event : str
#         Event to be analyzed.
#     bands : list
#         List of bands to be analyzed. The default is ['theta', 'alpha', 'beta', 'slowgamma', 'hfa'].
#     win_len : int
#         Length of the window for the time-resolved regression. The default is 50.
#     step_size : int
#         Step size for the time-resolved regression. The default is 10.
#     nsurr : int
#         Number of surrogates to be run. The default is 500.
#     save_path : str
#         Base path to save the resulting dataframe. Saving, rather than returning, is handy for parallelized code across big data. 
#     do_save : bool
#         Whether to save or return the dataframe. 
#     """
    
#     pow_df = TFR_object.copy().pick_channels([chan_name]).to_data_frame()
#     # Rename the frequencies according to band 
#     pow_df['fband'] = pow_df.freq.apply(lambda x: 'theta' if x<10 else 'alpha' if (x>=10) & (x<14) else 'beta' if (x>=14) & (x<30) else 'slowgamma' if (x>=30) & (x<55) else 'hfa')
#     # Average across frequencies within a band, rename some columns 
#     tt_df = pow_df.groupby(['epoch', 'fband', 'time']).mean().reset_index().drop(columns=['freq']).rename(columns={'epoch':'trial', f'{chan_name}':'tfr'})
#     TFR_object.metadata['trial'] = tt_df['trial'].unique()
#     # Merge in task details 
#     tfr_def_freq = tt_df.merge(TFR_object.metadata, on=['trial'])
    
#     # TFR-specific setup (function is more general) 
#     band_regress_dfs = []

#     for fband in tfr_def_freq.fband.unique():
#         if fband not in bands:
#             continue
#         else:
#             ntrials = tfr_def_freq.trials.unique().shape[0]
#             nsamples = pow_df.time.unique().shape[0]
#             features = ['rpe', 'DPRIME']

#             analysis_df = tfr_def_freq[tfr_def_freq.fband==fband]
#             # trim the timeseries by one sample???
#             timeseries = analysis_df.tfr.values.reshape(ntrials, nsamples)
#             regressors = analysis_df[features].drop_duplicates()

#             regress_df = statistics_utils.time_resolved_regression(timeseries, regressors, win_len, step_size, do_zscore=True, sr=TFR_object.info['sfreq'])
#             all_surrs = statistics_utils.time_resolved_regression_perm(timeseries, regressors, win_len, step_size, do_zscore=True, sr=TFR_object.info['sfreq'], nsurr=nsurr)

#             merged_df = pd.merge(all_surrs, regress_df, on=['ts', 'sample'], suffixes=('_df2', '_df1'))

#             # Compute the p-values and correct them across time. 
#             # Note that I am conducting two-way tests because I am interested in both positive and negative regression betas
            
#             # Now let's correct ACROSS TIMEPOINTS. 
#             # TODO: implement multiple comparisons corrections across bands if analyzing multiple bands.
#             for feature in regressors.keys():

#                 # Count the number of entries for 'rpe' in df2 that exceed the value in df1
#                 p_upper= (merged_df[f'{feature}_df2'] > merged_df[f'{feature}_df1']).groupby(merged_df['sample']).sum()/nsurr
#                 _, p_upper_adjusted, _, _ = multitest.multipletests(p_upper, method='fdr_bh')
#                 p_lower= (merged_df[f'{feature}_df2'] < merged_df[f'{feature}_df1']).groupby(merged_df['sample']).sum()/nsurr
#                 _, p_lower_adjusted, _, _ = multitest.multipletests(p_lower, method='fdr_bh')

#                 regress_df[f'{feature}_p_upper_fdr'] = p_upper_adjusted
#                 regress_df[f'{feature}_p_lower_fdr'] = p_lower_adjusted

#             regress_df['fband'] = fband
#             band_regress_dfs.append(regress_df)

#     band_regress_df = pd.concat(band_regress_dfs)
#     band_regress_df['chan'] = chan_name
#     band_regress_df['region'] = elec_df[elec_df.label==chan_name].salman_region.values[0]
#     band_regress_df['subj'] = subj_id
    
#     if do_save:
#         # TODO: replace this with your own path structure.... 
#         band_regress_df.to_csv(f'{save_path}/{subj_id}/scratch/TFR/{event_name}/dfs/{chan_name}_time_regressed_surr.csv', index=False)
#     else: 
#         return band_regress_df
#     # print(f'done with subject {subj_id} channel {chan_name}')

    

