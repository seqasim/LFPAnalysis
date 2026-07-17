"""Archived / incomplete utility surfaces kept for one-release compatibility.

Callers should import from the original module names (which re-export thin
deprecation shims). Prefer the stable workflow API or the documented advanced
helpers instead of anything in this module.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _warn_archived(name: str, *, prefer: str | None = None) -> None:
    message = (
        f"`{name}` is archived in LFPAnalysis._scratch_utils and will be removed "
        "in a future release."
    )
    if prefer:
        message = f"{message} Prefer `{prefer}`."
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def _not_implemented(name: str, *, prefer: str) -> None:
    _warn_archived(name, prefer=prefer)
    raise NotImplementedError(
        f"`{name}` is not implemented. Prefer `{prefer}`."
    )


# ---------------------------------------------------------------------------
# Stubs (never had a working body)
# ---------------------------------------------------------------------------


def FOOOF_continuous(signal: np.ndarray) -> None:
    """Archived stub — continuous FOOOF was never implemented."""
    _not_implemented(
        "FOOOF_continuous",
        prefer="analysis_utils.FOOOF_compute_epochs or workflow.compute_spectral_features",
    )


def sliding_FOOOF(signal: np.ndarray) -> None:
    """Archived stub — sliding FOOOF was never implemented."""
    _not_implemented(
        "sliding_FOOOF",
        prefer="analysis_utils.FOOOF_compute_epochs or workflow.compute_spectral_features",
    )


def get_behav_ts(logfile: Any) -> None:
    """Archived stub — extract behavioral timestamps yourself, then sync."""
    _not_implemented(
        "get_behav_ts",
        prefer="sync_utils.synchronize_data with your own behavioral timestamps",
    )


def merge_multiple_ncs_files(ncs_files: Any) -> None:
    """Archived stub — multi-file NCS merge was never finished."""
    _not_implemented(
        "merge_multiple_ncs_files",
        prefer="nlx_utils.parse_subject_nlx_data / workflow.load_lfp",
    )


def rename_mne_channels(mne_data: Any, location_table_path: str) -> None:
    """Archived incomplete Iowa rename helper."""
    _not_implemented(
        "rename_mne_channels",
        prefer="iowa_utils.extract_names_connect_table + manual MNE rename",
    )


def laplacian_ref(
    mne_data: Any,
    elec_path: str,
    bad_channels: list,
    unmatched_seeg=None,
    site=None,
) -> None:
    """Archived stub — Laplacian referencing is not implemented."""
    _not_implemented(
        "laplacian_ref",
        prefer="ref_mne(..., method='wm'|'bipolar') or workflow.preprocess_lfp",
    )


# ---------------------------------------------------------------------------
# Unused live helpers moved here for soft archive
# ---------------------------------------------------------------------------


def getTimeFromFTmat(fname: str, var_name: str = "data") -> np.ndarray:
    """Load time vector from a FieldTrip .mat file (archived)."""
    import scipy.io as sio

    _warn_archived("getTimeFromFTmat", prefer="MNE time vectors / Epochs.times")
    data = sio.loadmat(fname)
    ft = data[var_name][0, 0]
    return np.asarray(ft["time"][0, 0]).squeeze()


def get_project_root() -> Path:
    """Archived helper that returned this package file path, not the repo root."""
    _warn_archived("get_project_root", prefer="Path(__file__).resolve().parents[...]")
    return Path(__file__).resolve()


def fit_permuted_model(y_permuted, X):
    """Archived OLS helper; logic lives inside permutation_regression_zscore."""
    import statsmodels.api as sm

    _warn_archived(
        "fit_permuted_model",
        prefer="statistics_utils.permutation_regression_zscore",
    )
    return sm.OLS(y_permuted, X).fit().params


def _swap_time_blocks(data: np.ndarray, cut_at: int) -> np.ndarray:
    """Archived single-cut swap; use _swap_time_blocks_batch instead."""
    _warn_archived(
        "_swap_time_blocks",
        prefer="oscillation_utils._swap_time_blocks_batch",
    )
    return np.concatenate([data[..., cut_at:], data[..., :cut_at]], axis=-1)


def hctsa_signal_features(signal: np.ndarray):
    """Archived catch22 feature wrapper."""
    from .validation import ensure_dependency

    _warn_archived(
        "hctsa_signal_features",
        prefer="call pycatch22 directly after pip install -e '.[analysis]'",
    )
    pycatch22 = ensure_dependency("pycatch22", install_hint="pip install -e '.[analysis]'")
    features = pycatch22.catch22_all(signal)
    return pd.DataFrame({"names": features["names"], "values": features["values"]})


def make_mne_scalp(load_path=None, overwrite: bool = True, return_data: bool = False):
    """Archived scalp-EDF loader (not used by the stable API)."""
    import re
    import mne
    from glob import glob

    _warn_archived("make_mne_scalp", prefer="workflow.load_lfp / make_mne for iEEG")
    edf_file = glob(f"{load_path}/*.edf")[0]
    mne_data = mne.io.read_raw_edf(edf_file, preload=True)
    pattern = re.compile(
        r"^(?:[FTCPOM][pz\d]?|AF\d?|FC\d?|CP\d?|PO\d?|TP\d?|FT\d?|OZ|Fp\d?)$",
        re.IGNORECASE,
    )

    def is_scalp_eeg_channel(name):
        base = re.split("[- ]", name)[0]
        return bool(pattern.match(base))

    scalp_channels = [ch for ch in mne_data.ch_names if is_scalp_eeg_channel(ch)]
    if not scalp_channels:
        raise ValueError(
            "No scalp EEG channels found in the data. Please check the channel names or the data format."
        )
    mne_data.pick_channels(scalp_channels)
    mne_data.info["line_freq"] = 60
    mne_data.notch_filter(freqs=(60, 120, 180, 240))
    if return_data:
        return mne_data
    return mne_data.save(f"{load_path}/scalp_raw.fif", overwrite=overwrite)

