# In this libray I want to write some functions that make certain statistical analyses easier to run 

import scipy as sp
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf


def time_resolved_regression(timeseries=None, regressors=None, win_len=200, slide_len=50): 
    """
    In this function, if you provide a 2D array of z-scored time-varying neural data and a sert of regressors, 
    this function will run a time-resolved generalized linear model with the provided regressor dataframe. 

    Typically, this timeseries will be HFA, and the default win_len and slide_len reflect this 

    timeseries: ndarray, trials x times 
    regressors: pandas df, index = trials, columns = regressors
    """

    # Smooth the timeseries 
    smoothed_data = np.zeros([timeseries.shape[0], (len(timeseries) / slide_len)])
    for trial in timeseries.shape[0]:
        smoothed_data[trial, :] = [np.nanmean(timeseries[:, i:i+win_len]) for i in range(0, len(timeseries), slide_len) if i+win_len <= len(timeseries)]

    # Run the regression
    models = []
    for ts in smoothed_data.shape[1]: 
        model_df = regressors.copy() 
        model_df['dv'] = smoothed_data[:, ts]
        formula = f'dv ~ 1+'+'+'.join(regressors.columns)
        mod = smf.glm(formula=formula, data=model_df, family=sm.families.Gaussian()).fit()
        models.append(mod)

    return models 

def time_resolved_regression_perm(timeseries=None, regressors=None, win_len=200, slide_len=50):

    models = time_resolved_regression(timeseries, regressors, win_len, slide_len, nsurr)

    # Generate permuted timeseries 
    shuffles = np.random.randint(1, len(timeseries), nsurr)
    all_surrs = []
    for surr in range(nsurr): 
        # shuffle in time
        surr_ts = np.roll(timeseries, shuffles[surr], axis=1)
        # shuffle trials 
        np.random.shuffle(surr_ts)
        # Re-run regression with permuted timeseries 
        surr_models = time_resolved_regression(surr_ts, regressors, win_len, slide_len)
        all_surrs.append(surr_models)

    # Detect significant timepoints 
    # TODO: 
    pass 

    

