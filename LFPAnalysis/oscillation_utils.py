import numpy as np
import pandas as pd
import numpy.matlib
import scipy.io as sio
from pathlib import Path
import statsmodels.api as sm
from scipy.stats.distributions import chi2
from mne_connectivity import phase_slope_index, spectral_connectivity_epochs, spectral_connectivity_time
import mne
from scipy.signal import hilbert
from mne.filter import next_fast_len
from IPython.display import clear_output
from joblib import delayed, Parallel

import scipy.special
import warnings
import scipy as sp


# Helper functions 

def find_nearest_value(array, value):
    """Find nearest value and index of float in array
    Parameters:
    array : Array of values [1d array]
    value : Value of interest [float]
    Returns:
    array[idx] : Nearest value [1d float]
    idx : Nearest index [1d float]
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def getTimeFromFTmat(fname, var_name='data'):
    """
    Get original timing from FieldTrip structure
    Solution based on https://github.com/mne-tools/mne-python/issues/2476
    """
    # load Matlab/Fieldtrip data
    mat = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
    ft_data = mat[var_name]
    # convert to mne
    n_trial = len(ft_data.trial)
    n_chans, n_time = ft_data.trial[0].shape
    #data = np.zeros((n_trial, n_chans, n_time))
    time = np.zeros((n_trial, n_time))
    for trial in range(n_trial):
        # data[trial, :, :] = ft_data.trial[trial]
        # Note that this indexes time_orig in the adapted structure
        time[trial, :] = ft_data.time_orig[trial]
    return time

def get_project_root() -> Path:
    return Path(__file__)
    
# def swap_time_blocks(data, random_state=None):

#     """Compute surrogates by swapping time blocks.
#     This function cuts the timeseries at a random time point. Then, both time
#     blocks are swapped.
#     Parameters
#     ----------
#     data : array_like
#         Array of shape (n_chan, ..., n_times).
#     random_state : int | None
#         Fix the random state of the machine for reproducible results.
#     Returns
#     -------
#     surr : array_like
#         Swapped timeseries to use to compute the distribution of
#         permutations
#     References
#     ----------
#     Source: https://www.sciencedirect.com/science/article/pii/S0959438814001640
#     """
    
#     if random_state is None:
#         random_state = int(np.random.randint(0, 10000, size=1))
#     rnd = np.random.RandomState(random_state)
    
#     # get the minimum / maximum shift
#     min_shift, max_shift = 1, None
#     if not isinstance(max_shift, (int, float)):
#         max_shift = data.shape[-1]
#     # random cutting point along time axis
#     cut_at = rnd.randint(min_shift, max_shift, (1,))
#     # split amplitude across time into two parts
#     surr = np.array_split(data, cut_at, axis=-1)
#     # revered elements
#     surr.reverse()
    
#     return np.concatenate(surr, axis=-1)

def make_surrogate_data(data, method='swap_epochs', n_shuffles=1000, rng_seed=42, return_generator=False):
    """Create surrogate data for a null hypothesis of connectivity.
    
    Parameters
    ----------
    data : mne.Epochs
        The data to be shuffled.
    method : str
        The method to use for shuffling. Options are 'swap_time_blocks' and
        'swap_epochs'.
    n_shuffles : int
        The number of shuffles to perform.
    rng_seed : int
        The random seed to use for the shuffling.
    return_generator : bool 
        If True, returns a generator object. If False, returns a list of surrogates.
    """
    if method =='swap_time_blocks':
        surrogate = _shuffle_within_epochs(data, n_shuffles, rng_seed)
    elif method == 'swap_epochs':
        surrogate = _shuffle_epochs(data, n_shuffles, rng_seed)
    if not return_generator:
        surrogate = [shuffle for shuffle in surrogate]
    return surrogate

def _shuffle_epochs(data, n_shuffles, rng_seed):
    """Shuffle epochs in data. 
    This function shuffles the order of the epochs (dim 0) separately for each channel (dim 1)"""
    data_arr = data.get_data(copy=True)
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_shuffles):
        surr_arr = np.zeros_like(data_arr)
        for ch_idx in range(data_arr.shape[1]):
            surr_arr[:, ch_idx] = rng.permutation(data_arr[:, ch_idx])
        new_epochs = mne.EpochsArray(surr_arr, info=data.info, verbose=False,
                              events = data.events, 
                              event_id = data.event_id)
        new_epochs.set_annotations(data.annotations)
        yield new_epochs

def _shuffle_within_epochs(data, n_shuffles, rng_seed):
    """Shuffle within epochs in data. 
    This function cuts the timeseries at a random time point. Then, both time
    blocks are swapped.
    """
    data_arr = data.get_data(copy=True)
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_shuffles):
        surr_arr = np.zeros_like(data_arr)
        cutpoints = rng.integers(1, data_arr.shape[-1], (data_arr.shape[0], data_arr.shape[1]))
        for ev_idx in range(data_arr.shape[0]):
            for ch_idx in range(data_arr.shape[1]):
                surr_arr[ev_idx, ch_idx] = _swap_time_blocks(data_arr[ev_idx, ch_idx], cutpoints[ev_idx, ch_idx])
        new_epochs = mne.EpochsArray(surr_arr, info=data.info, verbose=False,
                              events = data.events, 
                              event_id = data.event_id)
        new_epochs.set_annotations(data.annotations)
        yield new_epochs

def _swap_time_blocks(data, cut_at):
    """Swap time blocks in data at a given cutpoint."""
    surr = np.array_split(data, [cut_at], axis=-1)
    surr.reverse()
    return np.concatenate(surr, axis=-1)

def make_seed_target_df(elec_df, epochs, source_roi, target_roi): 
    
    """
    Create arrays of indices for mapping electrodes for connectivity analyses. Use the Epoch
    ch_names list itself to find the index of the electrode within the mne object 
    
    """
    
    seed_target_df = pd.DataFrame(columns=['seed', 'target'], index=['l', 'r'])

    for hemi in ['l', 'r']:
        source_ix = elec_df[(elec_df.hemisphere.str.lower()==hemi) & (elec_df.salman_region==source_roi)].label.values
        target_ix = elec_df[(elec_df.hemisphere.str.lower()==hemi) & (elec_df.salman_region==target_roi)].label.values
        
        seed_target_df['seed'][hemi] = []
        seed_target_df['target'][hemi] = []
        
        if (len(source_ix) > 0) & (len(target_ix) > 0):
            source_channels = mne.pick_channels(epochs.ch_names, source_ix)
            target_channels = mne.pick_channels(epochs.ch_names, target_ix)
            if isinstance(source_channels, list):
                seed_target_df['seed'][hemi].extend(source_channels)
            else:
                seed_target_df['seed'][hemi].append(source_channels)
            
            if isinstance(target_channels, list):
                seed_target_df['target'][hemi].extend(target_channels)
            else:
                seed_target_df['target'][hemi].append(target_channels)

    seed_target_df = seed_target_df[
                (seed_target_df['seed'].map(lambda d: len(d) > 0)) & (seed_target_df['target'].map(lambda d: len(d) > 0))]

    
    return seed_target_df


### 2/18/25: Add in gaussian-copula mutual information connectivity measures from Ince et al. 2017 
### Source: https://github.com/robince/gcmi/blob/master/python/gcmi.py
"""
Gaussian copula mutual information estimation
"""

def ctransform(x):
    """Copula transformation (empirical CDF)

    cx = ctransform(x) returns the empirical CDF value along the first
    axis of x. Data is ranked and scaled within [0 1] (open interval).

    """

    xi = np.argsort(np.atleast_2d(x))
    xr = np.argsort(xi)
    cx = (xr+1).astype(float) / (xr.shape[-1]+1)
    return cx
 

def copnorm(x):
    """Copula normalization
    
    cx = copnorm(x) returns standard normal samples with the same empirical
    CDF value as the input. Operates along the last axis.

    """
    #cx = sp.stats.norm.ppf(ctransform(x))
    cx = sp.special.ndtri(ctransform(x))
    return cx


def ent_g(x, biascorrect=True):
    """Entropy of a Gaussian variable in bits

    H = ent_g(x) returns the entropy of a (possibly 
    multidimensional) Gaussian variable x with bias correction.
    Columns of x correspond to samples, rows to dimensions/variables. 
    (Samples last axis)

    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    # demean data
    x = x - x.mean(axis=1)[:,np.newaxis]
    # covariance
    C = np.dot(x,x.T) / float(Ntrl - 1)
    chC = np.linalg.cholesky(C)

    # entropy in nats
    HX = np.sum(np.log(np.diagonal(chC))) + 0.5*Nvarx*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((Ntrl - np.arange(1,Nvarx+1).astype(float))/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HX = HX - Nvarx*dterm - psiterms.sum()

    # convert to bits
    return HX / ln2


def mi_gg(x, y, biascorrect=True, demeaned=False):
    """Mutual information (MI) between two Gaussian variables in bits
   
    I = mi_gg(x,y) returns the MI between two (possibly multidimensional)
    Gassian variables, x and y, with bias correction.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis) 
                                                                             
    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the esimtated MI.
    demeaned : false / true option (default false) which specifies whether th
    input data already has zero mean (true if it has been copula-normalized)

    """
    
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarxy = Nvarx+Nvary

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xy = np.vstack((x,y))
    if not demeaned:
        xy = xy - xy.mean(axis=1)[:,np.newaxis]
    Cxy = np.dot(xy,xy.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cx = Cxy[:Nvarx,:Nvarx]
    Cy = Cxy[Nvarx:,Nvarx:]

    chCxy = np.linalg.cholesky(Cxy)
    chCx = np.linalg.cholesky(Cx)
    chCy = np.linalg.cholesky(Cy)

    # entropies in nats
    # normalizations cancel for mutual information
    HX = np.sum(np.log(np.diagonal(chCx))) # + 0.5*Nvarx*(np.log(2*np.pi)+1.0)
    HY = np.sum(np.log(np.diagonal(chCy))) # + 0.5*Nvary*(np.log(2*np.pi)+1.0)
    HXY = np.sum(np.log(np.diagonal(chCxy))) # + 0.5*Nvarxy*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((Ntrl - np.arange(1,Nvarxy+1)).astype(float)/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HX = HX - Nvarx*dterm - psiterms[:Nvarx].sum()
        HY = HY - Nvary*dterm - psiterms[:Nvary].sum()
        HXY = HXY - Nvarxy*dterm - psiterms[:Nvarxy].sum()

    # MI in bits
    I = (HX + HY - HXY) / ln2
    return I


def gcmi_cc(x,y):
    """Gaussian-Copula Mutual Information between two continuous variables.

    I = gcmi_cc(x,y) returns the MI between two (possibly multidimensional)
    continuous variables, x and y, estimated via a Gaussian copula.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples first axis) 
    This provides a lower bound to the true MI value.

    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break
    for yi in range(Nvary):
        if (np.unique(y[yi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input y has more than 10% repeated values")
            break

    # copula normalization
    cx = copnorm(x)
    cy = copnorm(y)
    # parametric Gaussian MI
    I = mi_gg(cx,cy,True,True)
    return I


def mi_model_gd(x, y, Ym, biascorrect=True, demeaned=False):
    """Mutual information (MI) between a Gaussian and a discrete variable in bits
    based on ANOVA style model comparison.

    I = mi_model_gd(x,y,Ym) returns the MI between the (possibly multidimensional)
    Gaussian variable x and the discrete variable y.
    For 1D x this is a lower bound to the mutual information.
    Columns of x correspond to samples, rows to dimensions/variables. 
    (Samples last axis)
    y should contain integer values in the range [0 Ym-1] (inclusive).

    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the esimtated MI.
    demeaned : false / true option (default false) which specifies whether the
    input data already has zero mean (true if it has been copula-normalized)

    See also: mi_mixture_gd

    """

    x = np.atleast_2d(x)
    y = np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y should be an integer array")
    if not isinstance(Ym, int):
        raise ValueError("Ym should be an integer")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    if y.size != Ntrl:
        raise ValueError("number of trials do not match")

    if not demeaned:
        x = x - x.mean(axis=1)[:,np.newaxis]

    # class-conditional entropies
    Ntrl_y = np.zeros(Ym)
    Hcond = np.zeros(Ym)
    c = 0.5*(np.log(2.0*np.pi)+1)
    for yi in range(Ym):
        idx = y==yi
        xm = x[:,idx]
        Ntrl_y[yi] = xm.shape[1]
        xm = xm - xm.mean(axis=1)[:,np.newaxis]
        Cm = np.dot(xm,xm.T) / float(Ntrl_y[yi]-1)
        chCm = np.linalg.cholesky(Cm)
        Hcond[yi] = np.sum(np.log(np.diagonal(chCm))) # + c*Nvarx

    # class weights
    w = Ntrl_y / float(Ntrl)

    # unconditional entropy from unconditional Gaussian fit
    Cx = np.dot(x,x.T) / float(Ntrl-1)
    chC = np.linalg.cholesky(Cx)
    Hunc = np.sum(np.log(np.diagonal(chC))) # + c*Nvarx

    ln2 = np.log(2)
    if biascorrect:
        vars = np.arange(1,Nvarx+1)

        psiterms = sp.special.psi((Ntrl - vars).astype(float)/2.0) / 2.0
        dterm = (ln2 - np.log(float(Ntrl-1))) / 2.0
        Hunc = Hunc - Nvarx*dterm - psiterms.sum()

        dterm = (ln2 - np.log((Ntrl_y-1).astype(float))) / 2.0
        psiterms = np.zeros(Ym)
        for vi in vars:
            idx = Ntrl_y-vi
            psiterms = psiterms + sp.special.psi(idx.astype(float)/2.0)
        Hcond = Hcond - Nvarx*dterm - (psiterms/2.0)

    # MI in bits
    I = (Hunc - np.sum(w*Hcond)) / ln2
    return I


def gcmi_model_cd(x,y,Ym):
    """Gaussian-Copula Mutual Information between a continuous and a discrete variable
     based on ANOVA style model comparison.

    I = gcmi_model_cd(x,y,Ym) returns the MI between the (possibly multidimensional)
    continuous variable x and the discrete variable y.
    For 1D x this is a lower bound to the mutual information.
    Columns of x correspond to samples, rows to dimensions/variables.
    (Samples last axis)
    y should contain integer values in the range [0 Ym-1] (inclusive).

    See also: gcmi_mixture_cd

    """

    x = np.atleast_2d(x)
    y = np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y should be an integer array")
    if not isinstance(Ym, int):
        raise ValueError("Ym should be an integer")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    if y.size != Ntrl:
        raise ValueError("number of trials do not match")

    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break

    # check values of discrete variable
    if y.min()!=0 or y.max()!=(Ym-1):
        raise ValueError("values of discrete variable y are out of bounds")

    # copula normalization
    cx = copnorm(x)
    # parametric Gaussian MI
    I = mi_model_gd(cx,y,Ym,True,True)
    return I


def mi_mixture_gd(x, y, Ym):
    """Mutual information (MI) between a Gaussian and a discrete variable in bits
    calculated from a Gaussian mixture.

    I = mi_mixture_gd(x,y,Ym) returns the MI between the (possibly multidimensional)
    Gaussian variable x and the discrete variable y.
    Columns of x correspond to samples, rows to dimensions/variables.
    (Samples last axis)
    y should contain integer values in the range [0 Ym-1] (inclusive).

    See also: mi_model_gd

    """

    x = np.atleast_2d(x)
    y = np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y should be an integer array")
    if not isinstance(Ym, int):
        raise ValueError("Ym should be an integer")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    if y.size != Ntrl:
        raise ValueError("number of trials do not match")

    # class-conditional entropies
    Ntrl_y = np.zeros(Ym)
    Hcond = np.zeros(Ym)
    m = np.zeros((Ym,Nvarx))
    w = np.zeros(Ym)
    cc = 0.5*(np.log(2.0*np.pi)+1)
    C = np.zeros((Ym,Nvarx,Nvarx))
    chC = np.zeros((Ym,Nvarx,Nvarx))
    for yi in range(Ym):
        # class conditional data
        idx = y==yi
        xm = x[:,idx]
        # class mean
        m[yi,:] = xm.mean(axis=1)
        Ntrl_y[yi] = xm.shape[1]

        xm = xm - m[yi,:][:,np.newaxis]
        C[yi,:,:] = np.dot(xm,xm.T) / float(Ntrl_y[yi]-1)
        chC[yi,:,:] = np.linalg.cholesky(C[yi,:,:])
        Hcond[yi] = np.sum(np.log(np.diagonal(chC[yi,:,:]))) + cc*Nvarx

    # class weights
    w = Ntrl_y / float(Ntrl)

    # mixture entropy via unscented transform
    # See:
    # Huber, Bailey, Durrant-Whyte and Hanebeck
    # "On entropy approximation for Gaussian mixture random vectors"
    # http://dx.doi.org/10.1109/MFI.2008.4648062

    # Goldberger, Gordon, Greenspan
    # "An efficient image similarity measure based on approximations of 
    # KL-divergence between two Gaussian mixtures"
    # http://dx.doi.org/10.1109/ICCV.2003.1238387
    D = Nvarx
    Ds = np.sqrt(Nvarx)
    Hmix = 0.0
    for yi in range(Ym):
        Ps = Ds * chC[yi,:,:].T
        thsm = m[yi,:,np.newaxis]
        # unscented points for this class
        usc = np.hstack([thsm + Ps, thsm - Ps])

        # class log-likelihoods at unscented points
        log_lik = np.zeros((Ym,2*Nvarx))
        for mi in range(Ym):
            # demean points
            dx = usc -  m[mi,:,np.newaxis]
            # gaussian likelihood
            log_lik[mi,:] = _norm_innerv(dx, chC[mi,:,:]) - Hcond[mi] + 0.5*Nvarx

        # log mixture likelihood for these unscented points
        # sum over classes, axis=0
        logmixlik = sp.special.logsumexp(log_lik,axis=0,b=w[:,np.newaxis])

        # add to entropy estimate (sum over unscented points for this class)
        Hmix = Hmix + w[yi]*logmixlik.sum()

    Hmix = -Hmix / (2*D)

    # no bias correct
    I = (Hmix - np.sum(w*Hcond)) / np.log(2.0)
    return I

def _norm_innerv(x, chC):
    """ normalised innervations """
    m = np.linalg.solve(chC,x)
    w = -0.5 * (m * m).sum(axis=0)
    return w


def gcmi_mixture_cd(x,y,Ym):
    """Gaussian-Copula Mutual Information between a continuous and a discrete variable
    calculated from a Gaussian mixture.

    The Gaussian mixture is fit using robust measures of location (median) and scale
    (median absolute deviation) for each class.
    I = gcmi_mixture_cd(x,y,Ym) returns the MI between the (possibly multidimensional)
    continuous variable x and the discrete variable y.
    For 1D x this is a lower bound to the mutual information.
    Columns of x correspond to samples, rows to dimensions/variables.
    (Samples last axis)
    y should contain integer values in the range [0 Ym-1] (inclusive).

    See also: gcmi_model_cd

    """

    x = np.atleast_2d(x)
    y = np.squeeze(y)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    if y.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y should be an integer array")
    if not isinstance(Ym, int):
        raise ValueError("Ym should be an integer")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    if y.size != Ntrl:
        raise ValueError("number of trials do not match")

    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break

    # check values of discrete variable
    if y.min()!=0 or y.max()!=(Ym-1):
        raise ValueError("values of discrete variable y are out of bounds")

    # copula normalise each class
    # shift and rescale to match loc and scale of raw data
    # this provides a robust way to fit the gaussian mixture
    classdat = []
    ydat = []
    for yi in range(Ym):
        # class conditional data
        idx = y==yi
        xm = x[:,idx]
        cxm = copnorm(xm)

        xmmed = np.median(xm,axis=1)[:,np.newaxis]
        # robust measure of s.d. under Gaussian assumption from median absolute deviation
        xmmad = np.median(np.abs(xm - xmmed),axis=1)[:,np.newaxis]
        cxmscaled = cxm * (1.482602218505602*xmmad)
        # robust measure of loc from median
        cxmscaled = cxmscaled + xmmed
        classdat.append(cxmscaled)
        ydat.append(yi*np.ones(xm.shape[1],dtype=np.int))

    cx = np.concatenate(classdat,axis=1) 
    newy = np.concatenate(ydat)
    I = mi_mixture_gd(cx,newy,Ym)
    return I


def cmi_ggg(x, y, z, biascorrect=True, demeaned=False):
    """Conditional Mutual information (CMI) between two Gaussian variables
    conditioned on a third

    I = cmi_ggg(x,y,z) returns the CMI between two (possibly multidimensional)
    Gassian variables, x and y, conditioned on a third, z, with bias correction.
    If x / y / z are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis)

    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the esimtated MI.
    demeaned : false / true option (default false) which specifies whether the
    input data already has zero mean (true if it has been copula-normalized)

    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)
    if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
        raise ValueError("x, y and z must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarz = z.shape[0]
    Nvaryz = Nvary + Nvarz
    Nvarxy = Nvarx + Nvary
    Nvarxz = Nvarx + Nvarz
    Nvarxyz = Nvarx + Nvaryz

    if y.shape[1] != Ntrl or z.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xyz = np.vstack((x,y,z))
    if not demeaned:
        xyz = xyz - xyz.mean(axis=1)[:,np.newaxis]
    Cxyz = np.dot(xyz,xyz.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cz = Cxyz[Nvarxy:,Nvarxy:]
    Cyz = Cxyz[Nvarx:,Nvarx:]
    Cxz = np.zeros((Nvarxz,Nvarxz))
    Cxz[:Nvarx,:Nvarx] = Cxyz[:Nvarx,:Nvarx]
    Cxz[:Nvarx,Nvarx:] = Cxyz[:Nvarx,Nvarxy:]
    Cxz[Nvarx:,:Nvarx] = Cxyz[Nvarxy:,:Nvarx]
    Cxz[Nvarx:,Nvarx:] = Cxyz[Nvarxy:,Nvarxy:]

    chCz = np.linalg.cholesky(Cz)
    chCxz = np.linalg.cholesky(Cxz)
    chCyz = np.linalg.cholesky(Cyz)
    chCxyz = np.linalg.cholesky(Cxyz)

    # entropies in nats
    # normalizations cancel for cmi
    HZ = np.sum(np.log(np.diagonal(chCz))) # + 0.5*Nvarz*(np.log(2*np.pi)+1.0)
    HXZ = np.sum(np.log(np.diagonal(chCxz))) # + 0.5*Nvarxz*(np.log(2*np.pi)+1.0)
    HYZ = np.sum(np.log(np.diagonal(chCyz))) # + 0.5*Nvaryz*(np.log(2*np.pi)+1.0)
    HXYZ = np.sum(np.log(np.diagonal(chCxyz))) # + 0.5*Nvarxyz*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((Ntrl - np.arange(1,Nvarxyz+1)).astype(float)/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HZ = HZ - Nvarz*dterm - psiterms[:Nvarz].sum()
        HXZ = HXZ - Nvarxz*dterm - psiterms[:Nvarxz].sum()
        HYZ = HYZ - Nvaryz*dterm - psiterms[:Nvaryz].sum()
        HXYZ = HXYZ - Nvarxyz*dterm - psiterms[:Nvarxyz].sum()

    # MI in bits
    I = (HXZ + HYZ - HXYZ - HZ) / ln2
    return I


def gccmi_ccc(x,y,z):
    """Gaussian-Copula CMI between three continuous variables.

    I = gccmi_ccc(x,y,z) returns the CMI between two (possibly multidimensional)
    continuous variables, x and y, conditioned on a third, z, estimated via a
    Gaussian copula.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples first axis)

    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)
    if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
        raise ValueError("x, y and z must be at most 2d")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarz = z.shape[0]

    if y.shape[1] != Ntrl or z.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break
    for yi in range(Nvary):
        if (np.unique(y[yi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input y has more than 10% repeated values")
            break
    for zi in range(Nvarz):
        if (np.unique(z[zi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input y has more than 10% repeated values")
            break

    # copula normalization
    cx = copnorm(x)
    cy = copnorm(y)
    cz = copnorm(z)
    # parametric Gaussian CMI
    I = cmi_ggg(cx,cy,cz,True,True)
    return I


def gccmi_ccd(x,y,z,Zm):
    """Gaussian-Copula CMI between 2 continuous variables conditioned on a discrete variable.

    I = gccmi_ccd(x,y,z,Zm) returns the CMI between two (possibly multidimensional)
    continuous variables, x and y, conditioned on a third discrete variable z, estimated
    via a Gaussian copula.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples first axis)
    z should contain integer values in the range [0 Zm-1] (inclusive).

    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    if z.ndim > 1:
        raise ValueError("only univariate discrete variables supported")
    if not np.issubdtype(z.dtype, np.integer):
        raise ValueError("z should be an integer array")
    if not isinstance(Zm, int):
        raise ValueError("Zm should be an integer")

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]

    if y.shape[1] != Ntrl or z.size != Ntrl:
        raise ValueError("number of trials do not match")

    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break
    for yi in range(Nvary):
        if (np.unique(y[yi,:]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input y has more than 10% repeated values")
            break

    # check values of discrete variable
    if z.min()!=0 or z.max()!=(Zm-1):
        raise ValueError("values of discrete variable z are out of bounds")

    # calculate gcmi for each z value
    Icond = np.zeros(Zm)
    Pz = np.zeros(Zm)
    cx = []
    cy = []
    for zi in range(Zm):
        idx = z==zi
        thsx = copnorm(x[:,idx])
        thsy = copnorm(y[:,idx])
        Pz[zi] = idx.sum()
        cx.append(thsx)
        cy.append(thsy)
        Icond[zi] = mi_gg(thsx,thsy,True,True)

    Pz = Pz / float(Ntrl)

    # conditional mutual information
    CMI = np.sum(Pz*Icond)
    I = mi_gg(np.hstack(cx),np.hstack(cy),True,False)
    return (CMI,I)

def phase_gcmi(mne_data, seed_to_target, freqs0, freqs1=None):
    """
    
    Compute the gaussian-copula condition mutual information between the phase of two signals.
    Can be within-frequency or between-frequency coupling.

    Parameters
    ----------
    mne_data : epochs object
        MNE epochs object containing the data to be analyzed.
    seed_to_target : list of tuples 
        List of tuples containing the indices of the seed and target electrodes.
    freqs0 : list or tuple
        Frequency range for the first signal.
    freqs1 : list or tuple
        Frequency range for the second signal.

    """

    nevents = mne_data._data.shape[0]
    ntimes = mne_data._data.shape[-1] 
    nfft = next_fast_len(ntimes)  
    # npairs = len(seed_to_target[0])
    nsource = len(np.unique(seed_to_target[0]))
    ntarget = len(np.unique(seed_to_target[1]))

    if freqs1 is None: 
        # Assume within-frequency coupling
        freqs1 = freqs0
    
    signal0 = mne_data._data[:, np.unique(seed_to_target[0]), :]
    signal1 = mne_data._data[:, np.unique(seed_to_target[1]), :]

    signal0_filt = mne.filter.filter_data(signal0, 
                     mne_data.info['sfreq'], 
                     l_freq=freqs0[0], 
                     h_freq=freqs0[1])
    
    signal1_filt = mne.filter.filter_data(signal1,
                        mne_data.info['sfreq'],
                        l_freq=freqs0[0],
                        h_freq=freqs0[1])
    
    gcmi = []
    for ei in range(nevents):
        signal0_hilbert = hilbert(signal0_filt[ei, :, :], N=nfft, axis=-1)[..., :ntimes]
        signal0_phase = np.angle(signal0_hilbert)
        signal1_hilbert = hilbert(signal1_filt[ei, :, :], N=nfft, axis=-1)[..., :ntimes]
        signal1_phase = np.angle(signal1_hilbert)

        # compute gcmi for each source to all targets
        gcmi_mat = []
        for source_ix in range(nsource):
            for target_ix in range(ntarget): 
                signal0_phase_elec = np.squeeze(signal0_phase[source_ix, :])
                signal1_phase_elec = np.squeeze(signal1_phase[target_ix, :])
                I = gcmi_cc(signal0_phase_elec, signal1_phase_elec)
                gcmi_mat.append(I)
        
        gcmi.append(gcmi_mat)

    pairwise_connectivity = np.stack(gcmi) # size is (nevents, ntarget, nsource)

    return pairwise_connectivity

def amp_amp_coupling(mne_data, seed_to_target, freqs0, freqs1=None):
    """
    Compute the correlation between the amplitude envelope of two signals. 
    Can be within-frequency or between-frequency coupling.

    Parameters
    ----------
    mne_data : epochs object
        MNE epochs object containing the data to be analyzed.
    seed_to_target : list of tuples
        List of tuples containing the indices of the seed and target electrodes.
    freqs0 : list or tuple
        Frequency range for the first signal.
    freqs1 : list or tuple
        Frequency range for the second signal. If None, assume within-frequency coupling.

    Note: inspired by MNE's pairwise orthogonal envelope connectivity metric but altered for iEEG data 
    """

    nevents = mne_data._data.shape[0]
    ntimes = mne_data._data.shape[-1] 
    nfft = next_fast_len(ntimes)  
    # npairs = len(seed_to_target[0])
    nsource = len(np.unique(seed_to_target[0]))
    ntarget = len(np.unique(seed_to_target[1]))

    if freqs1 is None: 
        # Assume within-frequency coupling
        freqs1 = freqs0
    
    signal0 = mne_data._data[:, np.unique(seed_to_target[0]), :]
    signal1 = mne_data._data[:, np.unique(seed_to_target[1]), :]

    signal0_filt = mne.filter.filter_data(signal0, 
                     mne_data.info['sfreq'], 
                     l_freq=freqs0[0], 
                     h_freq=freqs0[1])
    
    signal1_filt = mne.filter.filter_data(signal1,
                        mne_data.info['sfreq'],
                        l_freq=freqs0[0],
                        h_freq=freqs0[1])
    
    corrs = []

    for ei in range(nevents):
        signal0_hilbert = hilbert(signal0_filt[ei, :, :], N=nfft, axis=-1)[..., :ntimes]
        signal0_amp = np.abs(signal0_hilbert)
        signal1_hilbert = hilbert(signal1_filt[ei, :, :], N=nfft, axis=-1)[..., :ntimes]
        signal1_amp = np.abs(signal1_hilbert)

        # Square and log the analytical amplitude: https://www.nature.com/articles/nn.3101#Sec15
        signal0_amp *= signal0_amp
        np.log(signal0, out=signal0)
        signal1_amp *= signal1_amp
        np.log(signal1, out=signal1)

        # subtract mean 
        signal0_amp_nomean = signal0_amp - np.mean(signal0_amp, axis=-1, keepdims=True)
        signal1_amp_nomean = signal1_amp - np.mean(signal1_amp, axis=-1, keepdims=True)

        # compute variances using linalg.norm (square, sum, sqrt) since mean=0
        signal0_amp_std = np.linalg.norm(signal0_amp_nomean, axis=-1)
        signal0_amp_std[signal0_amp_std == 0] = 1
        signal1_amp_std = np.linalg.norm(signal1_amp_nomean, axis=-1)
        signal1_amp_std[signal1_amp_std == 0] = 1

        # compute correlation for each source to all targets
        corr_mat = []
        for source_ix in range(nsource):
            for target_ix in range(ntarget): 
                signal0_amp_elec = np.squeeze(signal0_amp_nomean[source_ix, :])
                signal1_amp_elec = np.squeeze(signal1_amp_nomean[target_ix, :])
                corr = np.sum(signal1_amp_elec * signal0_amp_elec)
                corr /= signal0_amp_std[source_ix]
                corr /= signal1_amp_std[target_ix]
                corr_mat.append(corr)
                
        corrs.append(corr_mat)

    pairwise_connectivity = np.stack(corrs) # size is (nevents, ntarget, nsource)
    # reshape so all pairs are in order:


    return pairwise_connectivity

def compute_gc_tr(mne_data=None, 
                band=None,
                indices=None, 
                freqs=None, 
                n_cycles=None,
                rank=None, 
                gc_n_lags=15, 
                buf_ms=1000, 
                avg_over_dim='time'): 
    """
    Following https://mne.tools/mne-connectivity/stable/auto_examples/granger_causality.html#sphx-glr-auto-examples-granger-causality-py
    """

    indices_ab = (np.array([np.unique(indices[0]).tolist()]), np.array([np.unique(indices[1]).tolist()]))  # A => B
    indices_ba = (np.array([np.unique(indices[1]).tolist()]), np.array([np.unique(indices[0]).tolist()]))  # B => A
    
    if avg_over_dim == 'epochs':
        # compute Granger causality
        gc_ab = spectral_connectivity_epochs(
            mne_data,
            sfreq = mne_data.info['sfreq'],
            method=["gc"],
            indices=indices_ab,
            fmin=band[0], fmax=band[1],
            rank=rank,
            gc_n_lags=gc_n_lags,
            verbose='ERROR') 
        # A => B
        gc_ba = spectral_connectivity_epochs(
            mne_data,
            sfreq = mne_data.info['sfreq'],
            method=["gc"],
            indices=indices_ba,
            fmin=band[0], fmax=band[1],
            rank=rank,
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # B => A
                    
        # compute GC on time-reversed signals
        gc_tr_ab = spectral_connectivity_epochs(
            mne_data,
            sfreq = mne_data.info['sfreq'],        
            method=["gc_tr"],
            indices=indices_ab,
            fmin=band[0], fmax=band[1],
            rank=rank,
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # TR[A => B]

        gc_tr_ba = spectral_connectivity_epochs(
            mne_data,
            sfreq = mne_data.info['sfreq'],                
            method=["gc_tr"],
            indices=indices_ba,
            fmin=band[0], fmax=band[1],
            rank=rank,
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # TR[B => A]
    elif avg_over_dim =='time':
        # compute Granger causality
        gc_ab = spectral_connectivity_time(
            mne_data,
            sfreq = mne_data.info['sfreq'],
            method=["gc"],
            indices=indices_ab,
            fmin=band[0], fmax=band[1],
            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])],
            rank=rank,
            padding=(buf_ms / 1000), 
            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
            gc_n_lags=gc_n_lags,
            verbose='ERROR') 

        # A => B
        gc_ba = spectral_connectivity_time(
            mne_data,
            sfreq = mne_data.info['sfreq'],
            method=["gc"],
            indices=indices_ba,
            fmin=band[0], fmax=band[1],
            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])],
            rank=rank,
            padding=(buf_ms / 1000), 
            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # B => A
                    
        # compute GC on time-reversed signals
        gc_tr_ab = spectral_connectivity_time(
            mne_data,
            sfreq = mne_data.info['sfreq'],        
            method=["gc_tr"],
            indices=indices_ab,
            fmin=band[0], fmax=band[1],
            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])],
            rank=rank,
            padding=(buf_ms / 1000), 
            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # TR[A => B]

        gc_tr_ba = spectral_connectivity_time(
            mne_data,
            sfreq = mne_data.info['sfreq'],                
            method=["gc_tr"],
            indices=indices_ba,
            fmin=band[0], fmax=band[1],
            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])],
            rank=rank,
            padding=(buf_ms / 1000), 
            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # TR[B => A]

    net_gc = gc_ab.get_data() - gc_ba.get_data()  # [A => B] - [B => A]

    # compute net GC on time-reversed signals (TR[A => B] - TR[B => A])
    net_gc_tr = gc_tr_ab.get_data() - gc_tr_ba.get_data()

    # compute TRGC
    gc_tr = net_gc - net_gc_tr

    if avg_over_dim =='time':
        return gc_tr.mean(axis=-1)
    else:
        return np.squeeze(gc_tr)

def compute_surr_connectivity_epochs(surr_mne, indices, metric, band, freqs, n_cycles, surr_method = 'swap_epochs', rng_seed=None, gc_n_lags=15, buf_ms=1000):

    n_pairs = len(indices[0])
    # data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

    # surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 

    # for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
    #     surr_ch = swap_time_blocks(ch_dat, random_state=None)
    #     surr_dat[ix, :, :] = surr_ch

    # surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 


    # # make a new EpochArray from it
    # surr_mne = mne.EpochsArray(surr_dat, 
    #             mne_data.info, 
    #             tmin=mne_data.tmin, 
    #             events = mne_data.events, 
    #             event_id = mne_data.event_id,
    #             verbose='ERROR')

    # data = mne_data.get_data(copy=True)
    # surr_mne = make_surrogate_data(mne_data,
    # method=surr_method, n_shuffles=1, rng_seed=rng_seed, return_generator=False)



    if metric == 'psi':
        surr_conn = np.squeeze(phase_slope_index(surr_mne,
                                                    indices=indices,
                                                    sfreq=surr_mne.info['sfreq'],
                                                    mode='cwt_morlet',
                                                    fmin=band[0], fmax=band[1],
                                                    cwt_freqs=freqs,
                                                    cwt_n_cycles=n_cycles,
                                                    verbose='warning').get_data()[:, 0])
    elif metric == 'granger':
        # I don't want to compute multivariate GC, so refactor the indices: 
        surr_conn = []

        for ix, _ in enumerate(indices[0]):
            gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
        
            surr_gc = compute_gc_tr(mne_data=surr_mne, 
                    band=band,
                    indices=gc_indices, 
                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                    rank=None, 
                    gc_n_lags=gc_n_lags, 
                    buf_ms=buf_ms, 
                    avg_over_dim='epochs')
            
            surr_conn.append(surr_gc)
            
        surr_conn = np.vstack(surr_conn)
    else:
        surr_conn = np.squeeze(spectral_connectivity_epochs(surr_mne,
                                                        indices=indices,
                                                        method=metric,
                                                        sfreq=surr_mne.info['sfreq'],
                                                        mode='cwt_morlet',
                                                        fmin=band[0], fmax=band[1], faverage=True,
                                                        cwt_freqs=freqs,
                                                        cwt_n_cycles=n_cycles,
                                                       verbose='ERROR').get_data()[:, 0])
    if metric != 'granger':
        if n_pairs == 1:
            # reshape data
            surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))

        # crop the buffer now:
        buf_rs = int((buf_ms/1000) * surr_mne[0].info['sfreq'])
        surr_conn = surr_conn[:, buf_rs:-buf_rs]

    return surr_conn


def compute_surr_connectivity_time(surr_mne, indices, metric, band, freqs, n_cycles, buf_ms, surr_method = 'swap_epochs', rng_seed=42, gc_n_lags=15):

    n_pairs = len(indices[0])
    # data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

    # surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 

    # for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
    #     surr_ch = swap_time_blocks(ch_dat, random_state=None)
    #     surr_dat[ix, :, :] = surr_ch

    # surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 

    # # make a new EpochArray from it
    # surr_mne = mne.EpochsArray(surr_dat, 
    #             mne_data.info, 
    #             tmin=mne_data.tmin, 
    #             events = mne_data.events, 
    #             event_id = mne_data.event_id,
    #             verbose='ERROR')

    # data = mne_data.get_data(copy=True)
    # surr_mne = make_surrogate_data(mne_data,
    # method=surr_method, n_shuffles=1, rng_seed=rng_seed, return_generator=False)

    if metric == 'granger':
        # I don't want to compute multivariate GC, so refactor the indices: 
        surr_conn = []

        for ix, _ in enumerate(indices[0]):
            gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
        
            gc = compute_gc_tr(mne_data=surr_mne, 
                    band=band,
                    indices=gc_indices, 
                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                    rank=None, 
                    gc_n_lags=gc_n_lags, 
                    buf_ms=buf_ms, 
                    avg_over_dim='time')
            
            surr_conn.append(gc)
            
        surr_conn = np.hstack(surr_conn)
    else:
        surr_conn = np.squeeze(spectral_connectivity_time(data=surr_mne, 
                                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                                    average=False, 
                                    indices=indices, 
                                    method=metric, 
                                    sfreq=surr_mne.info['sfreq'], 
                                    mode='cwt_morlet', 
                                    fmin=band[0], fmax=band[1], faverage=True, 
                                    padding=(buf_ms / 1000), 
                                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                                    rank=None, 
                                    gc_n_lags=gc_n_lags,
                                    verbose='ERROR').get_data())
    
    if n_pairs == 1:
        # reshape data
        surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))

    return surr_conn


def compute_connectivity(mne_data=None, 
                        band=None,
                        metric=None, 
                        indices=None, 
                        freqs=None, 
                        n_cycles=None, 
                        buf_ms=1000, 
                        avg_over_dim='time',
                        surr_method = 'swap_epochs',
                        n_surr=500,
                        parallelize=False,
                        band1=None,
                        gc_n_lags=7):
    """
    Compute different connectivity metrics using mne.
    :param eeg_mne: MNE formatted EEG
    :param samplerate: sample rate of the data
    :param band: tuple of band of interest
    :param metric: 'psi' for directional, or for non_directional: ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', pli2_unbiased', 'dpli', 'wpli', 'wpli2_debiased']
    see: https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html
    :param indices: determine the source and target for connectivity. Matters most for directional metrics i.e. 'psi'
    :return:
    pairwise connectivity: array of pairwise weights for the connectivity metric with some number of timepoints
    """
    if metric == 'gr_tc':
        return (ValueError('Use the function compute_gc_tr'))

    elif metric in ['gc', 'imcoh']: 
        indices = (np.array([np.unique(indices[0]).tolist()]), np.array([np.unique(indices[1]).tolist()]))

    if avg_over_dim == 'epochs':
        if metric == 'amp': 
            return (ValueError('Cannot compute amplitude-amplitude coupling over epochs.'))
        if metric == 'psi': 
            pairwise_connectivity = np.squeeze(phase_slope_index(mne_data,
                                                                    indices=indices,
                                                                    sfreq=mne_data.info['sfreq'],
                                                                    mode='cwt_morlet',
                                                                    fmin=band[0], fmax=band[1],
                                                                    cwt_freqs=freqs,
                                                                    cwt_n_cycles=n_cycles,
                                                                    verbose='warning').get_data()[:, 0])
            # return pairwise_connectivity
        elif metric == 'granger':
            # I don't want to compute multivariate GC, so refactor the indices: 
            pairwise_connectivity = []

            for ix, _ in enumerate(indices[0]):
                gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
            
                gc = compute_gc_tr(mne_data=mne_data, 
                        band=band,
                        indices=gc_indices, 
                        freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                        n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                        rank=None, 
                        gc_n_lags=gc_n_lags, 
                        buf_ms=buf_ms, 
                        avg_over_dim='epochs')
                
                pairwise_connectivity.append(gc)
                
            pairwise_connectivity = np.vstack(pairwise_connectivity)

        else:
            pairwise_connectivity = np.squeeze(spectral_connectivity_epochs(mne_data,
                                                            indices=indices,
                                                            method=metric,
                                                            sfreq=mne_data.info['sfreq'],
                                                            mode='cwt_morlet',
                                                            fmin=band[0], fmax=band[1], faverage=True,
                                                            cwt_freqs=freqs,
                                                            cwt_n_cycles=n_cycles,
                                                            verbose='ERROR').get_data()[:, 0])
        if metric in ['gc', 'imcoh']:
            # no pairs here: computed over whole multivariate state space 
            n_pairs=1
        else: 
            n_pairs = len(indices[0])

        if metric != 'granger':
            if n_pairs == 1:
                # reshape data
                pairwise_connectivity = pairwise_connectivity.reshape((pairwise_connectivity.shape[0], n_pairs))
            # # crop the buffer now:
            buf_rs = int((buf_ms/1000) * mne_data.info['sfreq'])
            pairwise_connectivity = pairwise_connectivity[:, buf_rs:-buf_rs]

        if n_surr > 0:
            surr_mne = make_surrogate_data(mne_data,
                method=surr_method, n_shuffles=n_surr, return_generator=False)
                            
            if parallelize == True:
                def _process_surrogate_epochs(ns):
                    # print(f'Computing surrogate # {ns} - parallel')
                    surrogate_result = compute_surr_connectivity_epochs(surr_mne[ns], indices, metric, band, freqs, n_cycles, gc_n_lags=gc_n_lags, buf_ms=buf_ms)
                    return surrogate_result

                surrogates = Parallel(n_jobs=-1)(delayed(_process_surrogate_epochs)(ns) for ns in range(n_surr))
                surr_struct = np.stack(surrogates, axis=-1)
            else: 
                # data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

                surr_struct = np.zeros([pairwise_connectivity.shape[0], pairwise_connectivity.shape[1], n_surr]) # allocate space for all the surrogates 

                # progress_bar = tqdm(np.arange(n_surr), ascii=True, desc='Computing connectivity surrogates')
                # data = mne_data.get_data(copy=True)

                for ns in range(n_surr): 
                    # print(f'Computing surrogate # {ns}')
                    # surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 
                    # for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
                    #     surr_ch = swap_time_blocks(ch_dat, random_state=None)
                    #     surr_dat[ix, :, :] = surr_ch
                    # surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 
                    # # make a new EpochArray from it
                    # surr_mne = mne.EpochsArray(surr_dat, 
                    #             mne_data.info, 
                    #             tmin=mne_data.tmin, 
                    #             events = mne_data.events, 
                    #             event_id = mne_data.event_id)

                    # surr_mne = make_surrogate_data(mne_data.get_data(copy=False),
                    # method=surr_method, n_shuffles=1, return_generator=False)


                    if metric == 'psi':
                        surr_conn = np.squeeze(phase_slope_index(surr_mne[ns],
                                                                    indices=indices,
                                                                    sfreq=surr_mne[ns].info['sfreq'],
                                                                    mode='cwt_morlet',
                                                                    fmin=band[0], fmax=band[1],
                                                                    cwt_freqs=freqs,
                                                                    cwt_n_cycles=n_cycles,
                                                                    verbose='warning').get_data()[:, 0])
                    elif metric == 'granger':
                        # I don't want to compute multivariate GC, so refactor the indices: 
                        surr_conn = []

                        for ix, _ in enumerate(indices[0]):
                            gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
                        
                            surr_gc = compute_gc_tr(mne_data=surr_mne[ns], 
                                    band=band,
                                    indices=gc_indices, 
                                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                                    rank=None, 
                                    gc_n_lags=gc_n_lags, 
                                    buf_ms=buf_ms, 
                                    avg_over_dim='epochs')
                            
                            surr_conn.append(surr_gc)
                            
                        surr_conn = np.vstack(surr_conn)

                    else:
                        surr_conn = np.squeeze(spectral_connectivity_epochs(surr_mne[ns],
                                                                        indices=indices,
                                                                        method=metric,
                                                                        sfreq=surr_mne[ns].info['sfreq'],
                                                                        mode='cwt_morlet',
                                                                        fmin=band[0], fmax=band[1], faverage=True,
                                                                        cwt_freqs=freqs,
                                                                        cwt_n_cycles=n_cycles,
                                                                        verbose='ERROR').get_data()[:, 0])
                    if metric != 'granger':
                        if n_pairs == 1:
                            # reshape data
                            surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))
                        # crop the surrogate: 
                        surr_conn = surr_conn[:, buf_rs:-buf_rs]

                    surr_struct[:, :, ns] = surr_conn
                    clear_output(wait=True)

            surr_mean = np.nanmean(surr_struct, axis=-1)
            surr_std = np.nanstd(surr_struct, axis=-1)
            pairwise_connectivity = (pairwise_connectivity - surr_mean) / (surr_std)
            
            # surr_struct[:, :, -1] = pairwise_connectivity # add the real data in as the last entry 
            # z_struct = zscore(surr_struct, axis=-1) # take the zscore across surrogate runs and the real data 
            # pairwise_connectivity = z_struct[:, :, -1] # extract the real data
    elif avg_over_dim == 'time':    
        if metric == 'psi': 
            return (ValueError('Cannot compute psi over time.'))
        
        elif metric == 'gcmi':
            
            # crop the buffer first:
            buf_s = buf_ms / 1000
            mne_data.crop(tmin=mne_data.tmin + buf_s,
                          tmax=mne_data.tmax - buf_s)
            
            pairwise_connectivity = phase_gcmi(mne_data,
                                                indices,
                                                freqs0=band,
                                                freqs1=band1)
            n_pairs = len(indices[0])

            if n_pairs == 1:
                # reshape data
                pairwise_connectivity = pairwise_connectivity.reshape((pairwise_connectivity.shape[0], n_pairs))

            if n_surr > 0:

                surr_mne = make_surrogate_data(mne_data,
                method=surr_method, n_shuffles=n_surr, return_generator=False)        

                surr_struct = np.zeros([pairwise_connectivity.shape[0], pairwise_connectivity.shape[1], n_surr]) # allocate space for all the surrogates 

                for ns in range(n_surr): 

                    surr_conn = phase_gcmi(surr_mne[ns], 
                                                    indices, 
                                                    freqs0=band,
                                                    freqs1=band1)
                    if n_pairs == 1:
                        # reshape data
                        surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))

                    surr_struct[:, :, ns] = surr_conn
                    clear_output(wait=True)

                surr_mean = np.nanmean(surr_struct, axis=-1)
                surr_std = np.nanstd(surr_struct, axis=-1)
                pairwise_connectivity = (pairwise_connectivity - surr_mean) / (surr_std)

        elif metric == 'amp': 
            
            # crop the buffer first:
            buf_s = buf_ms / 1000
            mne_data.crop(tmin=mne_data.tmin + buf_s,
                          tmax=mne_data.tmax - buf_s)

            pairwise_connectivity = amp_amp_coupling(mne_data, 
                                                     indices, 
                                                     freqs0=band,
                                                     freqs1=band1)
            n_pairs = len(indices[0])

            if n_pairs == 1:
                # reshape data
                pairwise_connectivity = pairwise_connectivity.reshape((pairwise_connectivity.shape[0], n_pairs))

            if n_surr > 0:

                surr_mne = make_surrogate_data(mne_data,
                method=surr_method, n_shuffles=n_surr, return_generator=False)
                                
                # data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

                surr_struct = np.zeros([pairwise_connectivity.shape[0], pairwise_connectivity.shape[1], n_surr]) # allocate space for all the surrogates 

                # progress_bar = tqdm(np.arange(n_surr), ascii=True, desc='Computing connectivity surrogates')
                # data = mne_data.get_data(copy=True)


                for ns in range(n_surr): 
                    # print(f'Computing surrogate # {ns}')
                    # surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 
                    # for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
                    #     surr_ch = swap_time_blocks(ch_dat, random_state=None)
                    #     surr_dat[ix, :, :] = surr_ch
                    # surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 
                    # # make a new EpochArray from it
                    # surr_mne = mne.EpochsArray(surr_dat, 
                    #             mne_data.info, 
                    #             tmin=mne_data.tmin, 
                    #             events = mne_data.events, 
                    #             event_id = mne_data.event_id)

                    # surr_mne = make_surrogate_data(mne_data.get_data(copy=False),
                    # method=surr_method, n_shuffles=1, return_generator=False)

                    surr_conn = amp_amp_coupling(surr_mne[ns], 
                                                    indices, 
                                                    freqs0=band,
                                                    freqs1=band1)
                    if n_pairs == 1:
                        # reshape data
                        surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))

                    surr_struct[:, :, ns] = surr_conn
                    clear_output(wait=True)

                surr_mean = np.nanmean(surr_struct, axis=-1)
                surr_std = np.nanstd(surr_struct, axis=-1)
                pairwise_connectivity = (pairwise_connectivity - surr_mean) / (surr_std)
                # surr_struct[:, :, -1] = pairwise_connectivity # add the real data in as the last entry
                # z_struct = zscore(surr_struct, axis=-1) # take the zscore across surrogate runs and the real data
                # pairwise_connectivity = z_struct[:, :, -1] # extract the real data      
        else:
            if metric == 'granger':
                # I don't want to compute multivariate GC, so refactor the indices: 
                pairwise_connectivity = []

                for ix, _ in enumerate(indices[0]):
                    gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
                
                    gc = compute_gc_tr(mne_data=mne_data, 
                            band=band,
                            indices=gc_indices, 
                            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                            rank=None, 
                            gc_n_lags=gc_n_lags, 
                            buf_ms=buf_ms, 
                            avg_over_dim='time')
                    
                    pairwise_connectivity.append(gc)
                    
                pairwise_connectivity = np.hstack(pairwise_connectivity)
            else:
                pairwise_connectivity = np.squeeze(spectral_connectivity_time(data=mne_data, 
                                                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                                                    average=False, 
                                                    indices=indices, 
                                                    method=metric, 
                                                    sfreq=mne_data.info['sfreq'], 
                                                    mode='cwt_morlet', 
                                                    fmin=band[0], fmax=band[1], faverage=True, 
                                                    padding=(buf_ms / 1000), 
                                                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                                                    rank=None,
                                                    gc_n_lags=gc_n_lags,
                                                    verbose='ERROR').get_data())
                # This returns an array of shape (n_events, n_pairs) 
                # where n_pairs is the number of pairs of channels in indices
                # and n_events is the number of events in the data

            
            if metric in ['gc', 'imcoh']:
                # no pairs here: computed over whole multivariate state space 
                n_pairs=1
            else: 
                n_pairs = len(indices[0])

            if n_pairs == 1:
                # reshape data
                pairwise_connectivity = pairwise_connectivity.reshape((pairwise_connectivity.shape[0], n_pairs))

            if n_surr > 0:
                surr_mne = make_surrogate_data(mne_data,
                    method=surr_method, n_shuffles=n_surr, return_generator=False)
                                    
                if parallelize == True:
                    def _process_surrogate_time(ns):
                        # print(f'Computing surrogate # {ns} - parallel')
                        surrogate_result = compute_surr_connectivity_time(surr_mne[ns], indices, metric, band, freqs, n_cycles, buf_ms, gc_n_lags, rng_seed=ns)
                        return surrogate_result

                    surrogates = Parallel(n_jobs=-1)(delayed(_process_surrogate_time)(ns) for ns in range(n_surr))
                    surr_struct = np.stack(surrogates, axis=-1)
                else:
                    # data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

                    surr_struct = np.zeros([pairwise_connectivity.shape[0], pairwise_connectivity.shape[1], n_surr]) # allocate space for all the surrogates 

                    # progress_bar = tqdm(np.arange(n_surr), ascii=True, desc='Computing connectivity surrogates')
                    # data = mne_data.get_data(copy=True)


                    for ns in range(n_surr): 
                        # print(f'Computing surrogate # {ns}')
                        # surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 
                        # for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
                        #     surr_ch = swap_time_blocks(ch_dat, random_state=None)
                        #     surr_dat[ix, :, :] = surr_ch
                        # surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 
                        # # make a new EpochArray from it
                        # surr_mne = mne.EpochsArray(surr_dat, 
                        #             mne_data.info, 
                        #             tmin=mne_data.tmin, 
                        #             events = mne_data.events, 
                        #             event_id = mne_data.event_id)
                        
                        surr_conn = np.squeeze(spectral_connectivity_time(data=surr_mne[ns], 
                                                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                                                    average=False, 
                                                    indices=indices, 
                                                    method=metric, 
                                                    sfreq=surr_mne[ns].info['sfreq'], 
                                                    mode='cwt_morlet', 
                                                    fmin=band[0], fmax=band[1], faverage=True, 
                                                    padding=(buf_ms / 1000), 
                                                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                                                    gc_n_lags=gc_n_lags,
                                                    verbose='ERROR').get_data())
                        
                        if n_pairs == 1:
                            # reshape data
                            surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))

                        surr_struct[:, :, ns] = surr_conn
                        clear_output(wait=True)

                surr_mean = np.nanmean(surr_struct, axis=-1)
                surr_std = np.nanstd(surr_struct, axis=-1)
                pairwise_connectivity = (pairwise_connectivity - surr_mean) / (surr_std)
                # surr_struct[:, :, -1] = pairwise_connectivity # add the real data in as the last entry
                # z_struct = zscore(surr_struct, axis=-1) # take the zscore across surrogate runs and the real data
                # pairwise_connectivity = z_struct[:, :, -1] # extract the real data            

    return pairwise_connectivity



########################################################################################


"""
BOSC (Better Oscillation Detection) function library
Rewritten from MATLAB to Python by Julian Q. Kosciessa

The original license information follows:
---
This file is part of the Better OSCillation detection (BOSC) library.

The BOSC library is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The BOSC library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2010 Jeremy B. Caplan, Adam M. Hughes, Tara A. Whitten
and Clayton T. Dickson.
---
"""

def BOSC_tf(eegsignal,F,Fsample,wavenumber):
    """
    Computes the Better Oscillation Detection (BOSC) time-frequency matrix for a given LFP signal.

    Args:
    - eegsignal (numpy.ndarray): The LFP signal to compute the BOSC time-frequency matrix for.
    - F (numpy.ndarray): The frequency range to compute the BOSC time-frequency matrix over.
    - Fsample (float): The sampling frequency of the LFP signal.
    - wavenumber (float): The wavenumber to use for the Morlet wavelet.

    Returns:
    - B (numpy.ndarray): The BOSC time-frequency matrix.
    - T (numpy.ndarray): The time vector corresponding to the BOSC time-frequency matrix.
    - F (numpy.ndarray): The frequency vector corresponding to the BOSC time-frequency matrix.
    """

    st=1./(2*np.pi*(F/wavenumber))
    A=1./np.sqrt(st*np.sqrt(np.pi))
    # initialize the time-frequency matrix
    B = np.zeros((len(F),len(eegsignal)))
    B[:] = np.nan
    # loop through sampled frequencies
    for f in range(len(F)):
        #print(f)
        t=np.arange(-3.6*st[f],(3.6*st[f]),1/Fsample)
        # define Morlet wavelet
        m=A[f]*np.exp(-t**2/(2*st[f]**2))*np.exp(1j*2*np.pi*F[f]*t)
        y=np.convolve(eegsignal,m, 'full')
        y=abs(y)**2
        B[f,:]=y[np.arange(int(np.ceil(len(m)/2))-1, len(y)-int(np.floor(len(m)/2)), 1)]
        T=np.arange(1,len(eegsignal)+1,1)/Fsample
    return B, T, F


def BOSC_detect(b,powthresh,durthresh,Fsample):
    """
    detected=BOSC_detect(b,powthresh,durthresh,Fsample)
    This function detects oscillations based on a wavelet power
    timecourse, b, a power threshold (powthresh) and duration
    threshold (durthresh) returned from BOSC_thresholds.m.
    
    It now returns the detected vector which is already episode-detected.
    
    b - the power timecourse (at one frequency of interest)
    
    durthresh - duration threshold in  required to be deemed oscillatory
    powthresh - power threshold
    
    returns:
    detected - a binary vector containing the value 1 for times at
               which oscillations (at the frequency of interest) were
               detected and 0 where no oscillations were detected.
    
    note: Remember to account for edge effects by including
    "shoulder" data and accounting for it afterwards!
    
    To calculate Pepisode:
    Pepisode=length(find(detected))/(length(detected));
    """                           

    # number of time points
    nT=len(b)
    #t=np.arange(1,nT+1,1)/Fsample
    
    # Step 1: power threshold
    x=b>powthresh
    # we have to turn the boolean to numeric
    x = np.array(list(map(np.int, x)))
    # show the +1 and -1 edges
    dx=np.diff(x)
    if np.size(np.where(dx==1))!=0:
        pos=np.where(dx==1)[0]+1
        #pos = pos[0]
    else: pos = []
    if np.size(np.where(dx==-1))!=0:
        neg=np.where(dx==-1)[0]+1
        #neg = neg[0]
    else: neg = []

    # now do all the special cases to handle the edges
    detected=np.zeros(b.shape)
    if not any(pos) and not any(neg):
        # either all time points are rhythmic or none
        if all(x==1):
            H = np.array([[0],[nT]])
        elif all(x==0):
            H = np.array([])
    elif not any(pos):
        # i.e., starts on an episode, then stops
        H = np.array([[0],neg])
        #np.concatenate(([1],neg), axis=0)
    elif not any(neg):
        # starts, then ends on an ep.
        H = np.array([pos,[nT]])
        #np.concatenate((pos,[nT]), axis=0)
    else:
        # special-case, create the H double-vector
        if pos[0]>neg[0]:
            # we start with an episode
            pos = np.append(0,pos)
        if neg[-1]<pos[-1]:
            # we end with an episode
            neg = np.append(neg,nT)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        H = np.array([pos,neg])
        #np.concatenate((pos,neg), axis=0)
    
    if H.shape[0]>0: 
        # more than one "hole"
        # find epochs lasting longer than minNcycles*period
        goodep=H[1,]-H[0,]>=durthresh
        if not any(goodep):
            H = [] 
        else: 
            H = H[:,goodep.nonzero()][:,0]
            # mark detected episode on the detected vector
            for h in range(H.shape[1]):
                detected[np.arange(H[0][h], H[1][h],1)]=1
        
    # ensure that outputs are integer
    detected = np.array(list(map(np.int, detected)))
    return detected

def eBOSC_getThresholds(cfg_eBOSC, TFR, eBOSC):
    """This function estimates the static duration and power thresholds and
    saves information regarding the overall spectrum and background.
    Inputs: 
               cfg | config structure with cfg.eBOSC field
               TFR | time-frequency matrix
               eBOSC | main eBOSC output structure; will be updated
    
    Outputs: 
               eBOSC   | updated w.r.t. background info (see below)
                       | bg_pow: overall power spectrum
                       | bg_log10_pow: overall power spectrum (log10)
                       | pv: intercept and slope of fit
                       | mp: linear background power
                       | pt: power threshold
               pt | empirical power threshold
               dt | duration threshold
    """

    # concatenate power estimates in time across trials of interest
    
    trial2extract = cfg_eBOSC['trial_background']
    # remove BGpad at beginning and end to avoid edge artifacts
    time2extract = np.arange(cfg_eBOSC['pad.background_sample'], TFR.shape[2]-cfg_eBOSC['pad.background_sample'],1)
    # index both trial and time dimension simultaneously
    TFR = TFR[np.ix_(trial2extract,range(TFR.shape[1]),time2extract)]
    # concatenate trials in time dimension: permute dimensions, then reshape
    TFR_t = np.transpose(TFR, [1,2,0])
    BG = TFR_t.reshape(TFR_t.shape[0],TFR_t.shape[1]*TFR_t.shape[2])
    del TFR_t, trial2extract, time2extract    
    # plt.imshow(BG[:,0:100], extent=[0, 1, 0, 1])
    
    # if frequency ranges should be exluded to reduce the influence of
    # rhythmic peaks on the estimation of the linear background, the
    # following section removes these specified ranges
    freqKeep = np.ones(cfg_eBOSC['F'].shape, dtype=bool)
    # allow for no peak removal
    if cfg_eBOSC['threshold.excludePeak'].size == 0:
        print("NOT removing frequency peaks from the background")
    else:
        print("Removing frequency peaks from the background")
        # n-dimensional arrays allow for the removal of multiple peaks
        for indExFreq in range(cfg_eBOSC['threshold.excludePeak'].shape[0]):
            # find empirical peak in specified range
            freqInd1 = np.where(cfg_eBOSC['F'] >= cfg_eBOSC['threshold.excludePeak'][indExFreq,0])[0][0]
            freqInd2 = np.where(cfg_eBOSC['F'] <= cfg_eBOSC['threshold.excludePeak'][indExFreq,1])[-1][-1]
            freqidx = np.arange(freqInd1,freqInd2+1)
            meanbg_within_range = list(BG[freqidx,:].mean(1))
            indPos = meanbg_within_range.index(max(meanbg_within_range))
            indPos = freqidx[indPos]
            # approximate wavelet extension in frequency domain
            # note: we do not remove the specified range, but the FWHM
            # around the empirical peak
            LowFreq = cfg_eBOSC['F'][indPos]-(((2/cfg_eBOSC['wavenumber'])*cfg_eBOSC['F'][indPos])/2)
            UpFreq = cfg_eBOSC['F'][indPos]+(((2/cfg_eBOSC['wavenumber'])*cfg_eBOSC['F'][indPos])/2)
            # index power estimates within the above range to remove from BG fit
            freqKeep[np.logical_and(cfg_eBOSC['F'] >= LowFreq, cfg_eBOSC['F'] <= UpFreq)] = False

    fitInput = {}
    fitInput['f_'] = cfg_eBOSC['F'][freqKeep]
    fitInput['BG_'] = BG[freqKeep, :]
   
    dataForBG = np.log10(fitInput['BG_']).mean(1)
    
    # perform the robust linear fit, only including putatively aperiodic components (i.e., peak exclusion)
    # replicate TukeyBiweight from MATLABs robustfit function
    exog = np.log10(fitInput['f_'])
    exog = sm.add_constant(exog)
    endog = dataForBG
    rlm_model = sm.RLM(endog, exog, M=sm.robust.norms.TukeyBiweight())
    rlm_results = rlm_model.fit()
    # MATLAB: b = robustfit(np.log10(fitInput['f_']),dataForBG)
    pv = np.zeros(2)
    pv[0] = rlm_results.params[1]
    pv[1] = rlm_results.params[0]
    mp = 10**(np.polyval(pv,np.log10(cfg_eBOSC['F'])))

    # compute eBOSC power (pt) and duration (dt) thresholds: 
    # power threshold is based on a chi-square distribution with df=2 and mean as estimated above
    pt=chi2.ppf(cfg_eBOSC['threshold.percentile'],2)*mp/2
    # duration threshold is the specified number of cycles, so it scales with frequency
    dt=(cfg_eBOSC['threshold.duration']*cfg_eBOSC['fsample']/cfg_eBOSC['F'])
    dt=np.transpose(dt, [1,0])

    # save multiple time-invariant estimates that could be of interest:
    # overall wavelet power spectrum (NOT only background)
    time2encode = np.arange(cfg_eBOSC['pad.total_sample'], BG.shape[1]-cfg_eBOSC['pad.total_sample'],1)
    eBOSC['static.bg_pow'].loc[cfg_eBOSC['tmp_channel'],:] = BG[:,time2encode].mean(1)
    # eBOSC[cfg_eBOSC['tmp_channelID']] = {'static.bg_pow': BG[:,time2encode].mean(1)}
    # log10-transformed wavelet power spectrum (NOT only background)
    eBOSC['static.bg_log10_pow'].loc[cfg_eBOSC['tmp_channel'],:] = np.log10(BG[:,time2encode]).mean(1)
    # intercept and slope parameters of the robust linear 1/f fit (log-log)
    eBOSC['static.pv'].loc[cfg_eBOSC['tmp_channel'],:] = pv
    # linear background power at each estimated frequency
    eBOSC['static.mp'].loc[cfg_eBOSC['tmp_channel'],:] = mp
    # statistical power threshold
    eBOSC['static.pt'].loc[cfg_eBOSC['tmp_channel'],:] = pt

    return eBOSC, pt, dt

def eBOSC_episode_sparsefreq(cfg_eBOSC, detected, TFR):
    """Sparsen the detected matrix along the frequency dimension
    """    
    # print('Creating sparse detected matrix ...')
    
    freqWidth = (2/cfg_eBOSC['wavenumber'])*cfg_eBOSC['F']
    lowFreq = cfg_eBOSC['F']-(freqWidth/2)
    highFreq = cfg_eBOSC['F']+(freqWidth/2)
    # %% define range for each frequency across which max. is detected
    fmat = np.zeros([cfg_eBOSC['F'].shape[0],3])
    for [indF,valF] in enumerate(cfg_eBOSC['F']):
        #print(indF)
        lastVal = np.where(cfg_eBOSC['F']<=lowFreq[indF])[0]
        if len(lastVal)>0:
            # first freq falling into range
            fmat[indF,0] = lastVal[-1]+1
        else: fmat[indF,0] = 0
        firstVal = np.where(cfg_eBOSC['F']>=highFreq[indF])[0]
        if len(firstVal)>0:
            # last freq falling into range
            fmat[indF,2] = firstVal[0]-1
        else: fmat[indF,2] = cfg_eBOSC['F'].shape[0]-1
    fmat[:,1] = np.arange(0, cfg_eBOSC['F'].shape[0],1)
    del indF
    range_cur = np.diff(fmat, axis=1)
    range_cur = [int(np.max(range_cur[:,0])), int(np.max(range_cur[:,1]))]
    # %% perform the actual search
    # initialize variables
    # append frequency search space (i.e. range at both ends. first index refers to lower range
    c1 = np.zeros([int(range_cur[0]),TFR.shape[1]])
    c2 = TFR*detected
    c3 = np.zeros([int(range_cur[1]),TFR.shape[1]])
    tmp_B = np.concatenate([c1, c2, c3])
    del c1,c2,c3
    # preallocate matrix (incl. padding , which will be removed)
    detected = np.zeros(tmp_B.shape)
    # loop across frequencies. note that indexing respects the appended segments
    freqs_to_search = np.arange(int(range_cur[0]), int(tmp_B.shape[0]-range_cur[1]),1)
    for f in freqs_to_search:
        # encode detected positions where power is higher than in LOWER and HIGHER ranges
        range1 = [f+np.arange(1,int(range_cur[1])+1)][0]
        range2 = [f-np.arange(1,int(range_cur[0])+1)][0]
        ranges = np.concatenate([range1,range2])
        detected[f,:] = np.logical_and(tmp_B[f,:] != 0, np.min(tmp_B[f,:] >= tmp_B[ranges,:],axis=0))
    # only retain data without padded zeros
    detected = detected[freqs_to_search,:]
    return detected

def eBOSC_episode_postproc_fwhm(cfg_eBOSC, episodes, TFR):
    """
    % This function performs post-processing of input episodes by checking
    % whether 'detected' time points can trivially be explained by the FWHM of
    % the wavelet used in the time-frequency transform.
    %
    % Inputs: 
    %           cfg | config structure with cfg.eBOSC field
    %           episodes | table of episodes
    %           TFR | time-frequency matrix
    %
    % Outputs: 
    %           episodes_new | updated table of episodes
    %           detected_new | updated binary detected matrix
    """
    
    print("Applying FWHM post-processing ...")
    
    # re-initialize detected_new (for post-proc results)
    detected_new = np.zeros(TFR.shape)
    # initialize new dictionary to save results in
    episodesTable = {}
    for entry in episodes:
        episodesTable[entry] = []

    for e in range(len(episodes['Trial'])):
        # get temporary frequency vector
        f_ = episodes['Frequency'][e]
        f_unique = np.unique(f_)           
        # find index within minor tolerance (float arrays)
        f_ind_unique = np.where(np.abs(cfg_eBOSC['F'][:,None] - f_unique) < 1e-5)
        f_ind_unique = f_ind_unique[0]
        # get temporary amplitude vector
        a_ = episodes['Power'][e]
        # location in time with reference to matrix TFR
        t_ind = np.int_(np.arange(episodes['ColID'][e][0], episodes['ColID'][e][-1]+1))
        # initiate bias matrix (only requires to encode frequencies occuring within episode)
        biasMat = np.zeros([len(f_unique),len(a_)])

        for tp in range(len(a_)):
            # The FWHM correction is done independently at each
            # frequency. To accomplish this, we actually reference
            # to the original data in the TF matrix.
            # search within frequencies that occur within the episode
            for f in range(len(f_unique)):
                # create wavelet with center frequency and amplitude at time point
                st=1/(2*np.pi*(f_unique/cfg_eBOSC['wavenumber']))
                step_size = 1/cfg_eBOSC['fsample']
                t=np.arange(-3.6*st[f],3.6*st[f]+step_size,step_size)
                wave = np.exp(-t**2/(2*st[f]**2))*np.exp(1j*2*np.pi*f_unique[f]*t)                
                if cfg_eBOSC['postproc.effSignal'] == 'all':
                    # Morlet wavelet with amplitude-power threshold modulation
                    m = TFR[f_ind_unique[f], int(t_ind[tp])]*wave
                elif cfg_eBOSC['postproc.effSignal'] == 'PT':
                    m = (TFR[f_ind_unique[f], int(t_ind[tp])]-
                         cfg_eBOSC['tmp.pt'][f_ind_unique[f]])*wave
                # amplitude of wavelet
                wl_a = abs(m)
                maxval = max(wl_a)
                maxloc = np.where(np.abs(wl_a[:,None] - maxval) < 1e-5)[0][0]
                index_fwhm = np.where(wl_a>= maxval/2)[0][0]
                # amplitude at fwhm, freq
                fwhm_a = wl_a[index_fwhm]
                if cfg_eBOSC['postproc.effSignal'] =='PT':
                    # re-add power threshold
                    fwhm_a = fwhm_a+cfg_eBOSC['tmp.pt'][f_ind_unique[f]]
                correctionDist = maxloc-index_fwhm
                # extract FWHM amplitude of frequency- and amplitude-specific wavelet
                # check that lower fwhm is part of signal 
                if tp-correctionDist >= 0:
                    # and that existing value is lower than update
                    if biasMat[f,tp-correctionDist] < fwhm_a:
                        biasMat[f,tp-correctionDist] = fwhm_a
                # check that upper fwhm is part of signal 
                if tp+correctionDist+1 <= biasMat.shape[1]:
                    # and that existing value is lower than update
                    if biasMat[f,tp+correctionDist] < fwhm_a:
                        biasMat[f,tp+correctionDist] = fwhm_a

        # plt.imshow(biasMat, extent=[0, 1, 0, 1])

        # retain only those points that are larger than the FWHM
        aMat_retain = np.zeros(biasMat.shape)
        indFreqs = np.where(np.abs(f_[:,None] - f_unique) < 1e-5)
        indFreqs = indFreqs[1]
        for indF in range(len(f_unique)):
            aMat_retain[indF,np.where(indFreqs == indF)[0]] = np.transpose(a_[indFreqs == indF])
        # anything that is lower than the convolved wavelet is removed
        aMat_retain[aMat_retain <= biasMat] = 0

        # identify which time points to retain and discard
        # Options: only correct at signal edge; correct within entire signal
        keep = aMat_retain.mean(0)>0
        keep = keep>0
        if cfg_eBOSC['postproc.edgeOnly'] == 'yes':
            keepEdgeRemovalOnly = np.zeros([len(keep)],dtype=bool)
            keepEdgeRemovalOnly[np.arange(np.where(keep==1)[0][0],np.where(keep==1)[0][-1]+1)] = True
            keep = keepEdgeRemovalOnly
            del keepEdgeRemovalOnly
            
        # get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))
    
        if max(d_keep) == 1 and min(d_keep) == -1:
            # start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0]-1
            for i in range(len(ind_epsd_begin)):
                # check for passing the duration requirement
                # get average frequency
                tmp_col = np.arange(ind_epsd_begin[i],ind_epsd_end[i]+1)
                avg_frq = np.mean(f_[tmp_col])
                # match to closest frequency
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC['F'], avg_frq)
                # check number of data points to fulfill number of cycles criterion
                num_pnt = np.floor((cfg_eBOSC['fsample']/ avg_frq) * int(np.reshape(cfg_eBOSC['threshold.duration'],[-1,1])[indF]))
                # if duration criterion remains fulfilled, encode in table
                if len(tmp_col) >= num_pnt:
                    # update all data in table with new episode limits
                    episodesTable['RowID'].append(episodes['RowID'][e][tmp_col])
                    episodesTable['ColID'].append([t_ind[tmp_col[0]], t_ind[tmp_col[-1]]])
                    episodesTable['Frequency'].append(f_[tmp_col])
                    episodesTable['FrequencyMean'].append(np.mean(episodesTable['Frequency'][-1]))
                    episodesTable['Power'].append(a_[tmp_col])
                    episodesTable['PowerMean'].append(np.mean(episodesTable['Power'][-1]))
                    episodesTable['DurationS'].append(np.diff(episodesTable['ColID'][-1])[0] / cfg_eBOSC['fsample'])
                    episodesTable['DurationC'].append(episodesTable['DurationS'][-1] * episodesTable['FrequencyMean'][-1])
                    episodesTable['Trial'].append(cfg_eBOSC['tmp_trial'])
                    episodesTable['Channel'].append(cfg_eBOSC['tmp_channel']) 
                    episodesTable['Onset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][0])])
                    episodesTable['Offset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][-1])])
                    episodesTable['SNR'].append(episodes['SNR'][e][tmp_col])
                    episodesTable['SNRMean'].append(np.mean(episodesTable['SNR'][-1]))
                    # set all detected points to one in binary detected matrix
                    detected_new[episodesTable['RowID'][-1],t_ind[tmp_col]] = 1
                    
    # plt.imshow(detected_new, extent=[0, 1, 0, 1])
    # return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new

def eBOSC_episode_postproc_maxbias(cfg_eBOSC, episodes, TFR):
    """
    % This function performs post-processing of input episodes by checking
    % whether 'detected' time points can be explained by the simulated extension of
    % the wavelet used in the time-frequency transform.
    %
    % Inputs: 
    %           cfg | config structure with cfg.eBOSC field
    %           episodes | table of episodes
    %           TFR | time-frequency matrix
    %
    % Outputs: 
    %           episodes_new | updated table of episodes
    %           detected_new | updated binary detected matrix
    
    % This method works as follows: we estimate the bias introduced by
    % wavelet convolution. The bias is represented by the amplitudes
    % estimated for the zero-shouldered signal (i.e. for which no real 
    % data was initially available). The influence of episodic
    % amplitudes on neighboring time points is assessed by scaling each
    % time point's amplitude with the last 'rhythmic simulated time
    % point', i.e. the first time wavelet amplitude in the simulated
    % rhythmic time points. At this time point the 'bias' is maximal,
    % although more precisely, this amplitude does not represent a
    % bias per se.
    """
    
    print("Applying maxbias post-processing ...")
    
    # re-initialize detected_new (for post-proc results)
    N_freq = TFR.shape[0]
    N_tp = TFR.shape[1]
    detected_new = np.zeros([N_freq, N_tp]);
    # initialize new dictionary to save results in
    # this is required as episodes may split, thus needing novel entries
    episodesTable = {}
    for entry in episodes:
        episodesTable[entry] = []
    
    # generate "bias" matrix
    # the logic here is as follows: we take a sinusoid, zero-pad it, and get the TFR
    # the bias is the tfr power produced for the padding (where power should be zero)
    B_bias = np.zeros([len(cfg_eBOSC['F']),len(cfg_eBOSC['F']),2*N_tp+1])
    amp_max = np.zeros([len(cfg_eBOSC['F']), len(cfg_eBOSC['F'])])
    for f in range(len(cfg_eBOSC['F'])):
        # temporary time vector and signal
        step_size = 1/cfg_eBOSC['fsample']
        time = np.arange(step_size, 1/cfg_eBOSC['F'][f]+step_size,step_size)
        tmp_sig = np.cos(time*2*np.pi*cfg_eBOSC['F'][f])*-1+1
        # signal for time-frequency analysis
        signal = np.concatenate((np.zeros([N_tp]), tmp_sig, np.zeros([N_tp])))
        [tmp_bias_mat, tmp_time, tmp_freq] = BOSC_tf(signal,cfg_eBOSC['F'],cfg_eBOSC['fsample'],cfg_eBOSC['wavenumber'])
        # bias matrix
        points_begin = np.arange(0,N_tp+1)
        points_end = np.arange(N_tp,B_bias.shape[2]+1)
        # for some reason, we have to transpose the matrix here, as the submatrix dimension order changes???
        B_bias[f,:,points_begin] = np.transpose(tmp_bias_mat[:,points_begin])
        B_bias[f,:,points_end] = np.transpose(np.fliplr(tmp_bias_mat[:,points_begin]))
        # maximum amplitude
        amp_max[f,:] = B_bias[f,:,:].max(1)
        # plt.imshow(amp_max, extent=[0, 1, 0, 1])

    # midpoint index
    ind_mid = N_tp+1
    # loop episodes
    for e in range(len(episodes['Trial'])):
        # get temporary frequency vector
        f_ = episodes['Frequency'][e]
        # get temporary amplitude vector
        a_ = episodes['Power'][e]
        m_ = np.zeros([len(a_),len(a_)])
        # location in time with reference to matrix TFR
        t_ind = np.arange(int(episodes['ColID'][e][0]),int(episodes['ColID'][e][-1]+1))
        # indices of time points' frequencies within "bias" matrix
        f_vec = episodes['RowID'][e]
        # figure; hold on;
        for tp in range(len(a_)):
            # index of current point's frequency within "bias" matrix
            ind_f = f_vec[tp]
            # get bias vector that varies with frequency of the
            # timepoints in the episode
            temporalBiasIndices = np.arange(ind_mid+1-tp,ind_mid+len(a_)-tp+1)
            ind1 = numpy.matlib.repmat(ind_f,len(f_vec),1)
            ind2 = np.reshape(f_vec,[-1,1])
            ind3 = np.reshape(temporalBiasIndices,[-1,1])
            indices = np.ravel_multi_index([ind1, ind2, ind3], 
                                           dims = B_bias.shape, order = 'C')
            tmp_biasVec = B_bias.flatten('C')[indices]
            # temporary "bias" vector (frequency-varying)
            if cfg_eBOSC['postproc.effSignal'] == 'all':
                tmp_bias = ((tmp_biasVec/np.reshape(amp_max[ind_f,f_vec],[-1,1]))*a_[tp])
            elif cfg_eBOSC['postproc.effSignal'] == 'PT':
                tmp_bias = ((tmp_biasVec/np.reshape(amp_max[ind_f,f_vec],[-1,1]))*
                            (a_[tp]-cfg_eBOSC['tmp.pt'][ind_f])) + cfg_eBOSC['tmp.pt'][ind_f]
            # compare to data
            m_[tp,:] = np.transpose(a_ >= tmp_bias)
            #plot(a_', 'k'); hold on; plot(tmp_bias, 'r');

        # identify which time points to retain and discard
        # Options: only correct at signal edge; correct within entire signal
        keep = m_.sum(0) == len(a_)
        if cfg_eBOSC['postproc.edgeOnly'] == 'yes':
            # keep everything that would be kept within the vector,
            # no removal within episode except for edges possible
            keepEdgeRemovalOnly = np.zeros([len(keep)],dtype=bool)
            keepEdgeRemovalOnly[np.arange(np.where(keep==1)[0][0],np.where(keep==1)[0][-1]+1)] = True
            keep = keepEdgeRemovalOnly
            del keepEdgeRemovalOnly

        # get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))
    
        if max(d_keep) == 1 and min(d_keep) == -1:
            # start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0]-1
            for i in range(len(ind_epsd_begin)):
                # check for passing the duration requirement
                # get average frequency
                tmp_col = np.arange(ind_epsd_begin[i],ind_epsd_end[i]+1)
                avg_frq = np.mean(f_[tmp_col])
                # match to closest frequency
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC['F'], avg_frq)
                # check number of data points to fulfill number of cycles criterion
                num_pnt = np.floor((cfg_eBOSC['fsample']/ avg_frq) * int(np.reshape(cfg_eBOSC['threshold.duration'],[-1,1])[indF]))
                # if duration criterion remains fulfilled, encode in table
                if len(tmp_col) >= num_pnt:
                    # update all data in table with new episode limits
                    episodesTable['RowID'].append(episodes['RowID'][e][tmp_col])
                    episodesTable['ColID'].append([t_ind[tmp_col[0]], t_ind[tmp_col[-1]]])
                    episodesTable['Frequency'].append(f_[tmp_col])
                    episodesTable['FrequencyMean'].append(np.mean(episodesTable['Frequency'][-1]))
                    episodesTable['Power'].append(a_[tmp_col])
                    episodesTable['PowerMean'].append(np.mean(episodesTable['Power'][-1]))
                    episodesTable['DurationS'].append(np.diff(episodesTable['ColID'][-1])[0] / cfg_eBOSC['fsample'])
                    episodesTable['DurationC'].append(episodesTable['DurationS'][-1] * episodesTable['FrequencyMean'][-1])
                    episodesTable['Trial'].append(cfg_eBOSC['tmp_trial'])
                    episodesTable['Channel'].append(cfg_eBOSC['tmp_channel']) 
                    episodesTable['Onset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][0])])
                    episodesTable['Offset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][-1])])
                    episodesTable['SNR'].append(episodes['SNR'][e][tmp_col])
                    episodesTable['SNRMean'].append(np.mean(episodesTable['SNR'][-1]))
                    # set all detected points to one in binary detected matrix
                    detected_new[episodesTable['RowID'][-1],t_ind[tmp_col]] = 1
    # return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new

def eBOSC_episode_rm_shoulder(cfg_eBOSC,detected1,episodes):
    """ Remove parts of the episode that fall into the 'shoulder' of individual
    trials. There is no check for adherence to a given duration criterion necessary,
    as the point of the padding of the detected matrix is exactly to account
    for allowing the presence of a few cycles.
    """

    print("Removing padding from detected episodes")

    ind1 = cfg_eBOSC['pad.detection_sample']
    ind2 = detected1.shape[1] - cfg_eBOSC['pad.detection_sample']
    rmv = []
    for j in range(len(episodes['Trial'])):
        # get time points of current episode
        tmp_col = np.arange(episodes['ColID'][j][0],episodes['ColID'][j][1]+1)
        # find time points that fall inside the padding (i.e. on- and offset)
        ex = np.where(np.logical_or(tmp_col < ind1, tmp_col >= ind2))[0]
        # remove padded time points from episodes
        tmp_col = np.delete(tmp_col, ex)
        episodes['RowID'][j] = np.delete(episodes['RowID'][j], ex)
        episodes['Power'][j] = np.delete(episodes['Power'][j], ex)
        episodes['Frequency'][j] = np.delete(episodes['Frequency'][j], ex)
        episodes['SNR'][j] = np.delete(episodes['SNR'][j], ex)
        # if nothing remains of episode: retain for later deletion
        if len(tmp_col)==0:
            rmv.append(j)
        else:
            # shift onset according to padding
            # Important: new col index is indexing w.r.t. to matrix AFTER
            # detected padding is removed!
            tmp_col = tmp_col - ind1
            episodes['ColID'][j] = [tmp_col[0], tmp_col[-1]]
            # re-compute mean frequency
            episodes['FrequencyMean'][j] = np.mean(episodes['Frequency'][j])
            # re-compute mean amplitude
            episodes['PowerMean'][j] = np.mean(episodes['Power'][j])
            # re-compute mean SNR
            episodes['SNRMean'][j] = np.mean(episodes['SNR'][j])
            # re-compute duration
            episodes['DurationS'][j] = np.diff(episodes['ColID'][j])[0] / cfg_eBOSC['fsample']
            episodes['DurationC'][j] = episodes['DurationS'][j] * episodes['FrequencyMean'][j]
            # update absolute on-/offsets (should remain the same)
            episodes['Onset'][j] = cfg_eBOSC['time.time_det'][int(episodes['ColID'][j][0])]
            episodes['Offset'][j] = cfg_eBOSC['time.time_det'][int(episodes['ColID'][j][-1])]
    # remove now empty episodes from table    
    for entry in episodes:
        # https://stackoverflow.com/questions/21032034/deleting-multiple-indexes-from-a-list-at-once-python
        episodes[entry] = [v for i, v in enumerate(episodes[entry]) if i not in rmv]
    return episodes

def eBOSC_episode_create(cfg_eBOSC,TFR,detected,eBOSC):
    """This function creates continuous rhythmic "episodes" and attempts to control for the impact of wavelet parameters.
      Time-frequency points that best represent neural rhythms are identified by
      heuristically removing temporal and frequency leakage. 
    
     Frequency leakage: at each frequency x time point, power has to exceed neighboring frequencies.
      Then it is checked whether the detected time-frequency points belong to
      a continuous episode for which (1) the frequency maximally changes by 
      +/- n steps (cfg.eBOSC.fstp) from on time point to the next and (2) that is at 
      least as long as n number of cycles (cfg.eBOSC.threshold.duration) of the average freqency
      of that episode (a priori duration threshold).
    
     Temporal leakage: The impact of the amplitude at each time point within a rhythmic episode on previous
      and following time points is tested with the goal to exclude supra-threshold time
      points that are due to the wavelet extension in time. 
    
    Input:   
               cfg         | config structure with cfg.eBOSC field
               TFR         | time-frequency matrix (excl. WLpadding)
               detected    | detected oscillations in TFR (based on power and duration threshold)
               eBOSC       | main eBOSC output structure; necessary to read in
                               prior eBOSC.episodes if they exist in a loop
    
    Output:  
               detected_new    | new detected matrix with frequency leakage removed
               episodesTable   | table with specific episode information:
                     Trial: trial index (corresponds to cfg.eBOSC.trial)
                     Channel: channel index
                     FrequencyMean: mean frequency of episode (Hz)
                     DurationS: episode duration (in sec)
                     DurationC: episode duration (in cycles, based on mean frequency)
                     PowerMean: mean amplitude of amplitude
                     Onset: episode onset in s
                     Offset: episode onset in s
                     Power: (cell) time-resolved wavelet-based amplitude estimates during episode
                     Frequency: (cell) time-resolved wavelet-based frequency
                     RowID: (cell) row index (frequency dimension): following eBOSC_episode_rm_shoulder relative to data excl. detection padding
                     ColID: (cell) column index (time dimension)
                     SNR: (cell) time-resolved signal-to-noise ratio: momentary amplitude/static background estimate at channel*frequency
                     SNRMean: mean signal-to-noise ratio
    """

    # initialize dictionary to save results in
    episodesTable = {}
    episodesTable['RowID'] = []
    episodesTable['ColID'] = []
    episodesTable['Frequency'] = []
    episodesTable['FrequencyMean'] = []
    episodesTable['Power'] = []
    episodesTable['PowerMean'] = []
    episodesTable['DurationS'] = []
    episodesTable['DurationC'] = []
    episodesTable['Trial'] = []
    episodesTable['Channel'] = []
    episodesTable['Onset'] = []
    episodesTable['Offset'] = []
    episodesTable['SNR'] = []
    episodesTable['SNRMean'] = []
    
    # %% Accounting for the frequency spread of the wavelet
    
    # Here, we compute the bandpass response as given by the wavelet
    # formula and apply half of the BP repsonse on top of the center frequency.
    # Because of log-scaling, the widths are not the same on both sides.
    
    detected = eBOSC_episode_sparsefreq(cfg_eBOSC, detected, TFR)    
    
    # %%  Create continuous rhythmic episodes
    
    # define step size in adjacency matrix
    cfg_eBOSC['fstp'] = 1
        
    # add zeros
    padding = np.zeros([cfg_eBOSC['fstp'],detected.shape[1]])
    detected_remaining = np.vstack([padding, detected, padding])
    detected_remaining[:,0] = 0
    detected_remaining[:,-1] = 0
    # detected_remaining serves as a dummy matrix; unless all entries from detected_remaining are
    # removed, we will continue extracting episodes
    tmp_B1 = np.vstack([padding, TFR*detected, padding])
    tmp_B1[:,0] = 0
    tmp_B1[:,-1] = 0
    detected_new = np.zeros(detected.shape)

    while sum(sum(detected_remaining)) > 0:
        # sampling point counter
        x = []
        y = []
        # find seed (remember that numpy uses row-first format!)
        # we need increasing x-axis sorting here
        [tmp_y,tmp_x] = np.where(np.matrix.transpose(detected_remaining)==1)
        x.append(tmp_x[0])
        y.append(tmp_y[0])
        # check next sampling point
        chck = 0
        while chck == 0:
            # next sampling point
            next_point = y[-1]+1
            next_freqs = np.arange(x[-1]-cfg_eBOSC['fstp'],
                          x[-1]+cfg_eBOSC['fstp']+1)
            tmp = np.where(detected_remaining[next_freqs,next_point]==1)[0]
            if tmp.size > 0:
                y.append(next_point)
                if tmp.size > 1:
                    # JQK 161017: It is possible that an episode is branching 
                    # two ways, hence we follow the 'strongest' branch; 
                    # Note that there is no correction for 1/f here, but 
                    # practically, it leads to satisfying results 
                    # (i.e. following the longer episodes).
                    tmp_data = tmp_B1[next_freqs,next_point]
                    tmp = np.where(tmp_data == max(tmp_data))[0]
                x.append(next_freqs[tmp[0]])
            else:
                chck = 1
            
        # check for passing the duration requirement
        # get average frequency
        avg_frq = np.mean(cfg_eBOSC['F'][np.array(x)-cfg_eBOSC['fstp']])
        # match to closest frequency
        [tmp_a, indF] = find_nearest_value(cfg_eBOSC['F'], avg_frq)
        # check number of data points to fulfill number of cycles criterion
        num_pnt = np.floor((cfg_eBOSC['fsample']/ avg_frq) * int(np.reshape(cfg_eBOSC['threshold.duration'],[-1,1])[indF]))
        if len(y) >= num_pnt:
            # %% encode episode that crosses duration threshold
            episodesTable['RowID'].append(np.array(x)-cfg_eBOSC['fstp'])
            episodesTable['ColID'].append([np.single(y[0]), np.single(y[-1])])
            episodesTable['Frequency'].append(np.single(cfg_eBOSC['F'][episodesTable['RowID'][-1]]))
            episodesTable['FrequencyMean'].append(np.single(avg_frq))
            tmp_x = episodesTable['RowID'][-1]
            tmp_y = np.arange(int(episodesTable['ColID'][-1][0]),int(episodesTable['ColID'][-1][1])+1)
            linIdx = np.ravel_multi_index([np.reshape(tmp_x,[-1,1]),
                                  np.reshape(tmp_y,[-1,1])], 
                                 dims=TFR.shape, order='C')
            episodesTable['Power'].append(np.single(TFR.flatten('C')[linIdx]))
            episodesTable['PowerMean'].append(np.mean(episodesTable['Power'][-1]))
            episodesTable['DurationS'].append(np.single(len(y)/cfg_eBOSC['fsample']))
            episodesTable['DurationC'].append(episodesTable['DurationS'][-1]*episodesTable['FrequencyMean'][-1])
            episodesTable['Trial'].append(cfg_eBOSC['tmp_trial']) # Note that the trial is non-zero-based
            episodesTable['Channel'].append(cfg_eBOSC['tmp_channel']) 
            # episode onset in absolute time
            episodesTable['Onset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][0])]) 
            # episode offset in absolute time
            episodesTable['Offset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][-1])]) 
            # extract (static) background power at frequencies
            episodesTable['SNR'].append(episodesTable['Power'][-1]/
                                 eBOSC['static.pt'].iloc[cfg_eBOSC['tmp_channelID'],
                                                         episodesTable['RowID'][-1]].values)
            episodesTable['SNRMean'].append(np.mean(episodesTable['SNR'][-1]))
            
            # remove processed segment from detected matrix
            detected_remaining[x,y] = 0
            # set all detected points to one in binary detected matrix
            rows = episodesTable['RowID'][-1]
            cols = np.arange(int(episodesTable['ColID'][-1][0]),
                                  int(episodesTable['ColID'][-1][1])+1)
            detected_new[rows,cols] = 1
        else:
            # %% remove episode from consideration due to being lower than duration
            detected_remaining[x,y] = 0
        
        # some sanity checks that episode selection was sensible
        #plt.imshow(detected, extent=[0, 1, 0, 1])
        #plt.imshow(detected_new, extent=[0, 1, 0, 1])
    
    # %%  Exclude temporal amplitude "leakage" due to wavelet smearing
    # temporarily pass on power threshold for easier access
    cfg_eBOSC['tmp.pt'] = eBOSC['static.pt'].loc[cfg_eBOSC['tmp_channel']].values
    
    # SQ note: This doesn't work too well so fuck it 
    # only do this if there are any episodes to fine-tune
    if cfg_eBOSC['postproc.use'] == 'yes' and len(episodesTable['Trial']) > 0:
        if cfg_eBOSC['postproc.method'] == 'FWHM':
            [episodesTable, detected_new] = eBOSC_episode_postproc_fwhm(cfg_eBOSC, episodesTable, TFR)
        elif cfg_eBOSC['postproc.method'] == 'MaxBias':
            [episodesTable, detected_new] = eBOSC_episode_postproc_maxbias(cfg_eBOSC, episodesTable, TFR)
        
    # %% remove episodes and part of episodes that fall into 'shoulder'
    
    if len(episodesTable['Trial']) > 0 and cfg_eBOSC['pad.detection_sample']>0:
        episodesTable = eBOSC_episode_rm_shoulder(cfg_eBOSC,detected_new,episodesTable)
    
    # %% if an episode list already exists, append results
    
    if 'episodes' in eBOSC:
        # initialize dictionary entries if not existing
        if not len(eBOSC['episodes']):
            for entry in episodesTable:
                eBOSC['episodes'][entry] = [] 
        # append current results
        for entry in episodesTable:
            episodesTable[entry] = eBOSC['episodes'][entry] + episodesTable[entry]
        
    return episodesTable, detected_new

def eBOSC_wrapper(cfg_eBOSC, data):
    """Main eBOSC wrapper function. Executes eBOSC subfunctions.
    Inputs: 
        cfg_eBOSC | dictionary containing the following entries:
            F                     | frequency sampling
            wavenumber            | wavelet family parameter (time-frequency tradeoff)
            fsample               | current sampling frequency of EEG data
            pad.tfr_s             | padding following wavelet transform to avoid edge artifacts in seconds (bi-lateral)
            pad.detection_s       | padding following rhythm detection in seconds (bi-lateral); 'shoulder' for BOSC eBOSC.detected matrix to account for duration threshold
            pad.total_s           | complete padding (WL + shoulder)
            pad.background_s      | padding of segments for BG (only avoiding edge artifacts)
            threshold.excludePeak | lower and upper bound of frequencies to be excluded during background fit (Hz) (previously: LowFreqExcludeBG HighFreqExcludeBG)
            threshold.duration    | vector of duration thresholds at each frequency (previously: ncyc)
            threshold.percentile  | percentile of background fit for power threshold
            postproc.use          | Post-processing of rhythmic eBOSC.episodes, i.e., wavelet 'deconvolution' (default = 'no')
            postproc.method       | Deconvolution method (default = 'MaxBias', FWHM: 'FWHM')
            postproc.edgeOnly     | Deconvolution only at on- and offsets of eBOSC.episodes? (default = 'yes')
            postproc.effSignal	  | Power deconvolution on whole signal or signal above power threshold? (default = 'PT')
            channel               | Subset of channels? (default: [] = all)
            trial                 | Subset of trials? (default: [] = all)
            trial_background      | Subset of trials for background? (default: [] = all)
        data | input time series data as a Pandas DataFrame: 
            - channels as columns
            - multiindex containing: 'time', 'epoch', 
    Outputs: 
        eBOSC | main eBOSC output dictionary containing the following entries:
            episodes | Dictionary: individual rhythmic episodes (see eBOSC_episode_create)
            detected | DataFrame: binary detected time-frequency points (prior to episode creation), pepisode = temporal average
            detected_ep | DataFrame: binary detected time-frequency points (following episode creation), abundance = temporal average
            cfg | config structure (see input)
    """

    # %% get list of channel names (very manual solution, replace if possible)

    channelNames = list(data.columns.values)
    channelNames.remove('time')
    channelNames.remove('condition')
    channelNames.remove('epoch')

    # %% define some defaults for included channels and trials, if not specified
    
    if not cfg_eBOSC['channel']:
        cfg_eBOSC['channel'] = channelNames # list of channel names
    
    if not cfg_eBOSC['trial']:
        # remember to count trial 1 as zero
        cfg_eBOSC['trial'] = list(np.arange(0,len(pd.unique(data['epoch']))))
    # else: # this ensures the zero count
    #     cfg_eBOSC['trial'] = list(np.array(cfg_eBOSC['trial']))
        
    if not cfg_eBOSC['trial_background']:
        cfg_eBOSC['trial_background'] = list(np.arange(0,len(pd.unique(data['epoch']))))
    # else: # this ensures the zero count
    #     cfg_eBOSC['trial_background'] = list(np.array(cfg_eBOSC['trial_background']) - 1)

    # %% calculate the sample points for paddding
    
    cfg_eBOSC['pad.tfr_sample'] = int(cfg_eBOSC['pad.tfr_s'] * cfg_eBOSC['fsample'])
    cfg_eBOSC['pad.detection_sample'] = int(cfg_eBOSC['pad.detection_s'] * cfg_eBOSC['fsample'])
    cfg_eBOSC['pad.total_s'] = cfg_eBOSC['pad.tfr_s'] + cfg_eBOSC['pad.detection_s']
    cfg_eBOSC['pad.total_sample'] = int(cfg_eBOSC['pad.tfr_sample'] + cfg_eBOSC['pad.detection_sample'])
    cfg_eBOSC['pad.background_sample'] = int(cfg_eBOSC['pad.tfr_sample'])
    
    # %% calculate time vectors (necessary for preallocating data frames)
    
    n_trial = len(cfg_eBOSC['trial'])
    n_freq = len(cfg_eBOSC['F'])
    n_time_total = len(pd.unique(data.loc[data['epoch']==0, ('time')]))
    # copy potentially non-continuous time values (assume that epoch is labeled 0)
    cfg_eBOSC['time.time_total'] = data.loc[data['epoch']==0, ('time')].values
    # alternatively: create a new time vector that is non-continuous and starts at zero
    # np.arange(0, 1/cfg_eBOSC['fsample']*(n_time_total) , 1/cfg_eBOSC['fsample'])
    # get timing and info for post-TFR padding removal
    tfr_time2extract = np.arange(cfg_eBOSC['pad.tfr_sample'], n_time_total-cfg_eBOSC['pad.tfr_sample'],1)
    cfg_eBOSC['time.time_tfr'] = cfg_eBOSC['time.time_total'][tfr_time2extract]
    n_time_tfr = len(cfg_eBOSC['time.time_tfr'])
    # get timing and info for post-detected padding removal
    det_time2extract = np.arange(cfg_eBOSC['pad.detection_sample'], n_time_tfr-cfg_eBOSC['pad.detection_sample'],1)
    cfg_eBOSC['time.time_det'] = cfg_eBOSC['time.time_tfr'][det_time2extract]
    n_time_det = len(cfg_eBOSC['time.time_det'])
        
    # %% preallocate data frames

    eBOSC = {}
    eBOSC['static.bg_pow'] = pd.DataFrame(columns=cfg_eBOSC['F'])
    eBOSC['static.bg_log10_pow'] = pd.DataFrame(columns=cfg_eBOSC['F'])    
    eBOSC['static.pv'] = pd.DataFrame(columns=['slope', 'intercept'])
    eBOSC['static.mp'] = pd.DataFrame(columns=cfg_eBOSC['F'])    
    eBOSC['static.pt'] = pd.DataFrame(columns=cfg_eBOSC['F'])   
    
    # Multiindex for channel x trial x frequency x time
    arrays = np.array([cfg_eBOSC['channel'],cfg_eBOSC['trial'],cfg_eBOSC['F'], cfg_eBOSC['time.time_det']],dtype=object)
    #tuples = list(zip(*arrays))
    names=["channel", "trial", "frequency", "time"]
    index=pd.MultiIndex.from_product(arrays,names=names)
    nullData=np.zeros(len(arrays[0]) * len(arrays[1]) * len(arrays[2]) * len(arrays[3]) )
    eBOSC['detected'] = pd.DataFrame(data=nullData, index=index)
    eBOSC['detected_ep'] = eBOSC['detected'].copy()
    del nullData, index
    
    eBOSC['episodes'] = {}

    # %% main eBOSC loop
    
    for channel in cfg_eBOSC['channel']:
        print('Channel: ' + channel + '; Nr. ' + str(cfg_eBOSC['channel'].index(channel)+1) + '/' + str(len(cfg_eBOSC['channel'])))
        cfg_eBOSC['tmp_channelID'] = cfg_eBOSC['channel'].index(channel)
        cfg_eBOSC['tmp_channel'] = channel
                
        # %% Step 1: time-frequency wavelet decomposition for whole signal to prepare background fit
        n_trial = len(cfg_eBOSC['trial'])
        n_freq = len(cfg_eBOSC['F'])
        n_time = len(pd.unique(data.loc[data['epoch']==0, ('time')]))
        TFR = np.zeros((n_trial, n_freq, n_time))
        TFR[:] = np.nan
        for trial in cfg_eBOSC['trial']:
            eegsignal = data.loc[data['epoch']==trial, (channel)]
            F = cfg_eBOSC['F']
            Fsample = cfg_eBOSC['fsample']
            wavenumber = cfg_eBOSC['wavenumber']
            [TFR[trial,:,:], tmp, tmp] = BOSC_tf(eegsignal,F,Fsample,wavenumber)
            del eegsignal,F,Fsample,wavenumber,tmp
            
        # %% plot example time-frequency spectrograms (only for intuition/debugging) 
        # assumes that multiple trials are present
        # plt.imshow(TFR[0,:,:], extent=[0, 1, 0, 1])
        # plt.imshow(TFR[:,:,:].mean(axis=0), extent=[0, 1, 0, 1])
        # plt.imshow(TFR[:,:,:].mean(axis=1), extent=[0, 1, 0, 1])
        # plt.imshow(TFR[:,:,:].mean(axis=2), extent=[0, 1, 0, 1])
                
        # %% Step 2: robust background power fit (see 2020 NeuroImage paper)
       
        [eBOSC, pt, dt] = eBOSC_getThresholds(cfg_eBOSC, TFR, eBOSC)
         
        # %% application of thresholds to single trials

        for trial in cfg_eBOSC['trial']:
            # print('Trial Nr. ' + str(trial+1) + '/' + str(len(cfg_eBOSC['trial'])))
            # encode current trial ID for later
            cfg_eBOSC['tmp_trialID'] = trial
            # trial ID in the intuitive convention
            cfg_eBOSC['tmp_trial'] = cfg_eBOSC['trial'].index(trial)+1

            # get wavelet transform for single trial
            # tfr padding is removed to avoid edge artifacts from the wavelet
            # transform. Note that a padding fpr detection remains attached so that there
            # is no problems with too few sample points at the edges to
            # fulfill the duration criterion.         
            time2extract = np.arange(cfg_eBOSC['pad.tfr_sample'], TFR.shape[2]-cfg_eBOSC['pad.tfr_sample'],1)
            TFR_ = np.transpose(TFR[trial,:,time2extract],[1,0])
            
            # %% Step 3: detect rhythms and calculate Pepisode
            # The next section applies both the power and the duration
            # threshold to detect individual rhythmic segments in the continuous signals.
            detected = np.zeros((TFR_.shape))
            for f in range(len(cfg_eBOSC['F'])):
                detected[f,:] = BOSC_detect(TFR_[f,:],pt[f],dt[f][0],cfg_eBOSC['fsample'])

            # remove padding for detection (matrix with padding required for refinement)
            time2encode = np.arange(cfg_eBOSC['pad.detection_sample'], detected.shape[1]-cfg_eBOSC['pad.detection_sample'],1)
            eBOSC['detected'].loc[(channel, trial)] = np.reshape(detected[:,time2encode],[-1,1])
            
            # %% Step 4 (optional): create table of separate rhythmic episodes
            [episodes, detected_ep] = eBOSC_episode_create(cfg_eBOSC,TFR_,detected,eBOSC)
            # insert detected episodes into episode structure
            eBOSC['episodes'] = episodes
            
            # remove padding for detection (already done for eBOSC.episodes)
            time2encode = np.arange(cfg_eBOSC['pad.detection_sample'], detected_ep.shape[1]-cfg_eBOSC['pad.detection_sample'],1)
            eBOSC['detected_ep'].loc[(channel, trial)] = np.reshape(detected_ep[:,time2encode],[-1,1])

            # %% Supplementary Plot: original eBOSC.detected vs. sparse episode power
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=2, ncols=1)
            # detected_cur = eBOSC['detected_ep'].loc[(channel, trial)]
            # detected_cur = detected_cur.pivot_table(index=['frequency'], columns='time')
            # curPlot = detected_cur*TFR_[:,time2encode]
            # axes[0].imshow(curPlot, aspect='auto', vmin = 0, vmax = 1)
            # detected_cur = eBOSC['detected'].loc[(channel, trial)]
            # detected_cur = detected_cur.pivot_table(index=['frequency'], columns='time')
            # curPlot = detected_cur*TFR_[:,time2encode]
            # axes[1].imshow(curPlot, aspect='auto', vmin = 0, vmax = 1)

    # %% return dictionaries back to caller script
    return eBOSC, cfg_eBOSC


def compute_eBOSC_parallel(chan_name, MNE_object, subj_id, elec_df, event_name, ev_dict, conditions, 
                           do_plot=False, save_path='/sc/arion/projects/guLab/Salman/EphysAnalyses', 
                           do_save=False, mean_across_time=False, mean_across_freqs=False, both_dfs=True, **kwargs):
    """

    This function is meant to parallelize our BOSC code to be computed over many channels simultaneously and save the results 
    to individual dataframes. 

    """

    if not os.path.exists(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs'):
        os.makedirs(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs')

    data_df = MNE_object.copy().pick_channels([chan_name]).to_data_frame(time_format=None)

    # parameters for eBOSC
    cfg_eBOSC = kwargs
    cfg_eBOSC['channel'] = [chan_name]

    # Compute BOSC: 
    [eBOSC, cfg] = eBOSC_wrapper(cfg_eBOSC, data_df)

    # Cut off buffer time
    if ev_dict[event_name][0] < 0:
        eBOSC['detected'] = eBOSC['detected'].query(f'(time>=0) & (time<={ev_dict[event_name][1]})')

    eBOSC['detected'] = eBOSC['detected'].reset_index().rename(columns={0:'prop_detect'})

    # Update: Let's actually do this AFTER loading the saved BOSC results so we are not being redundant. 
    # # Add events to the BOSC data:  
    # event_df['trial'] = eBOSC['detected']['trial'].unique()
    # eBOSC['detected'] = eBOSC['detected'].merge(event_df, on=['trial'])

    # identify frequency bands 
    eBOSC['detected']['fband'] = eBOSC['detected'].frequency.apply(lambda x: 'theta' if x<10 else 'alpha' if (x>=10) & (x<14) else 'beta' if (x>=14) & (x<30) else 'slowgamma' if (x>=30) & (x<55) else 'hfa')

    # # get rid of all the annoying line messages
    # clear_output(wait=True)

    # Dataframe for saving
    if both_dfs:
        time_averaged_df = pd.DataFrame(eBOSC['detected'].groupby(['trial', 'frequency']).mean()).reset_index().drop(columns=['time'])
        time_averaged_df.insert(0,'channel', chan_name)
        time_averaged_df.insert(0, 'region', elec_df[elec_df.label==chan_name].salman_region.values[0])
        time_averaged_df.insert(0,'subj', subj_id)
        time_averaged_df['event'] = event_name    
        if do_save: 
            time_averaged_df.to_csv(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs/{chan_name}_time_averaged_df.csv', index=False)

        time_resolved_df = eBOSC['detected'].groupby(['trial', 'fband', 'time']).mean().reset_index().drop(columns=['frequency'])
        if do_save: 
            time_resolved_df.to_csv(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs/{chan_name}_time_resolved_df.csv', index=False)
    
    if mean_across_time:
        time_averaged_df = pd.DataFrame(eBOSC['detected'].groupby(['trial', 'frequency']).mean()).reset_index().drop(columns=['time'])
        time_averaged_df.insert(0,'channel', chan_name)
        time_averaged_df.insert(0, 'region', elec_df[elec_df.label==chan_name].salman_region.values[0])
        time_averaged_df.insert(0,'subj', subj_id)
        time_averaged_df['event'] = event_name
        if do_save: 
            time_averaged_df.to_csv(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs/{chan_name}_time_averaged_df.csv', index=False)
    elif mean_across_freqs: 
        # Average across frequencies within a band, rename some columns 
        time_resolved_df = eBOSC['detected'].groupby(['trial', 'fband', 'time']).mean().reset_index().drop(columns=['frequency'])
        if do_save: 
            time_resolved_df.to_csv(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs/{chan_name}_time_resolved_df.csv', index=False)
    
#     THE FOLLOWING CODE APPLIE TO TFR parallelized code as well!! 
#     if do_plot:
#         # If we want to plot we need the event data and the condition information
#         fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[10,5], sharex=True, sharey=True)
#         for ix, cond in enumerate(conditions): 
#             # Plot: 
#             detected_avg = pd.DataFrame(eBOSC['detected'].query(cond).groupby(['frequency', 'time']).mean().drop(columns=['trial'])['prop_detect'])

#             # eBOSC['detected'].groupby(level=['frequency', 'time']).mean()
#             detected_avg = detected_avg.pivot_table(index=['frequency'], columns='time')
#             cur_multiindex = eBOSC['detected'].index
#             cur_time = eBOSC['detected']z.time.unique()
#             # cur_multiindex.get_level_values('time').unique()
#             cur_freq = eBOSC['detected'].frequency.unique()
#             # cur_multiindex.get_level_values('frequency').unique()

            
# #                 ax.vlines(250, 0, len(cfg_eBOSC['F']), 'white')
#             im = ax[ix].imshow(detected_avg, aspect = 'auto', interpolation='bicubic', cmap='rocket', vmin=0, vmax=.4)

#             [x0, x1] = ax[ix].get_xlim()
#             [y0, y1] = ax[ix].get_ylim()
#             xticks_loc = np.linspace(0,750, 4)
#             # [t for t in ax.get_xticks() if t>=x0 and t<=x1]
#             yticks_loc = [t for t in ax[ix].get_yticks() if t>=y1 and t<=y0]
#             x_label_list = np.round(cur_time[np.int_(xticks_loc)],1).tolist()
#             y_label_list = np.round(cur_freq[np.int_(yticks_loc)],1).tolist()
#             ax[ix].set_xticks(xticks_loc)
#             ax[ix].set_xticklabels(x_label_list)
#             ax[ix].set_yticks(yticks_loc)
#             ax[ix].set_yticklabels(y_label_list)
#             ax[ix].invert_yaxis()
#             ax[ix].set_xlabel('Time [s]')
#             ax[ix].set_ylabel('Frequency [Hz]') 
#             ax[ix].set_title(f'{cond}')
#             fig.colorbar(im, ax=ax[ix])
#         plt.suptitle('Avg. detected rhythms across trials', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/plots/{chan_name}_eBOSC.pdf', dpi=100)
#         plt.close()

# # USAGE example from: https://github.com/jkosciessa/eBOSC_py/blob/main/examples/eBOSC_example_empirical.ipynb
# pn = dict()
# pn['root']  = os.path.join(os.getcwd(),'..')
# pn['examplefile'] = os.path.join(pn['root'],'data','1160_rest_EEG_Rlm_Fhl_rdSeg_Art_EC.csv')
# pn['outfile'] = os.path.join(pn['root'],'data','example_out.npy')

# cfg_eBOSC = dict()
# cfg_eBOSC['F'] = 2 ** np.arange(1,6,.125)   # frequency sampling
# cfg_eBOSC['wavenumber'] = 6                 # wavelet parameter (time-frequency tradeoff)
# cfg_eBOSC['fsample'] = 500                  # current sampling frequency of EEG data
# cfg_eBOSC['pad.tfr_s'] = 1                  # padding following wavelet transform to avoid edge artifacts in seconds (bi-lateral)
# cfg_eBOSC['pad.detection_s'] = .5           # padding following rhythm detection in seconds (bi-lateral); 'shoulder' for BOSC eBOSC.detected matrix to account for duration threshold
# cfg_eBOSC['pad.background_s'] = 1           # padding of segments for BG (only avoiding edge artifacts)

# cfg_eBOSC['threshold.excludePeak'] = np.array([[8,15]])   # lower and upper bound of frequencies to be excluded during background fit (Hz) (previously: LowFreqExcludeBG HighFreqExcludeBG)
# cfg_eBOSC['threshold.duration'] = np.kron(np.ones((1,len(cfg_eBOSC['F']))),3) # vector of duration thresholds at each frequency (previously: ncyc)
# cfg_eBOSC['threshold.percentile'] = .95    # percentile of background fit for power threshold

# cfg_eBOSC['postproc.use'] = 'yes'           # Post-processing of rhythmic eBOSC.episodes, i.e., wavelet 'deconvolution' (default = 'no')
# cfg_eBOSC['postproc.method'] = 'FWHM'       # Deconvolution method (default = 'MaxBias', FWHM: 'FWHM')
# cfg_eBOSC['postproc.edgeOnly'] = 'yes'      # Deconvolution only at on- and offsets of eBOSC.episodes? (default = 'yes')
# cfg_eBOSC['postproc.effSignal'] = 'PT'      # Power deconvolution on whole signal or signal above power threshold

# cfg_eBOSC['channel'] = ['Oz']            # select posterior channels (default: all)
# cfg_eBOSC['trial'] = []                  # select trials (default: all, indicate in natural trial number (not zero-starting))
# cfg_eBOSC['trial_background'] = []       # select trials for background (default: all, indicate in natural trial

# # Either concatenate all epochs or use with continuous data: 
# [eBOSC, cfg] = eBOSC_wrapper(cfg_eBOSC, data)

# # Plot: 
# detected_avg = eBOSC['detected'].mean(level=['frequency', 'time'])
# detected_avg = detected_avg.pivot_table(index=['frequency'], columns='time')
# cur_multiindex = eBOSC['detected'].index
# cur_time = cur_multiindex.get_level_values('time').unique()
# cur_freq = cur_multiindex.get_level_values('frequency').unique()

# fig, ax = plt.subplots(nrows=1, ncols=1)
# im = ax.imshow(detected_avg, aspect = 'auto')
# [x0, x1] = ax.get_xlim()
# [y0, y1] = ax.get_ylim()
# xticks_loc = [t for t in ax.get_xticks() if t>=x0 and t<=x1]
# yticks_loc = [t for t in ax.get_yticks() if t>=y1 and t<=y0]
# x_label_list = np.round(cur_time[np.int_(xticks_loc)],1).tolist()
# y_label_list = np.round(cur_freq[np.int_(yticks_loc)],1).tolist()
# ax.set_xticks(xticks_loc)
# ax.set_xticklabels(x_label_list)
# ax.set_yticks(yticks_loc)
# ax.set_yticklabels(y_label_list)
# plt.colorbar(im, label='Proportion detected across trials')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.title('Avg. detected rhythms across trials', fontsize=12)
# plt.show()