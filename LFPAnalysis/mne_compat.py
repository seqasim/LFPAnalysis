"""Helpers for MNE operations that require float64 input arrays.

LFPAnalysis stores signal buffers as ``WORKING_DTYPE`` (float32) to save RAM.
MNE filtering (and related transforms such as Hilbert on filtered copies) require
real floating64 arrays. Use these helpers at every filter boundary so callers can
keep float32 storage without hitting MNE dtype errors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import WORKING_DTYPE

_WORKING_DTYPE = np.dtype(WORKING_DTYPE).type
_FILTER_DTYPE = np.float64


def upcast_mne_data(data: Any, dtype=_FILTER_DTYPE):
    """Ensure a preloaded MNE Raw/Epochs object uses ``dtype`` for filtering."""
    if hasattr(data, "preload") and not data.preload and hasattr(data, "load_data"):
        data.load_data()
    if not hasattr(data, "_data") or data._data is None:
        return data
    if data._data.dtype != dtype:
        data._data = np.asarray(data._data, dtype=dtype)
    return data


def downcast_mne_data(data: Any, dtype=None):
    """Cast preloaded MNE signal arrays back to the working dtype."""
    if dtype is None:
        dtype = _WORKING_DTYPE
    if not hasattr(data, "preload") or not data.preload:
        return data
    if not hasattr(data, "_data") or data._data is None:
        return data
    if data._data.dtype != dtype:
        data._data = np.asarray(data._data, dtype=dtype)
    return data


def filter_mne_object(data: Any, *args, restore_working_dtype: bool = True, **kwargs):
    """Copy ``data``, upcast to float64, run ``.filter``, optionally downcast.

    Set ``restore_working_dtype=False`` when the next step (e.g. Hilbert) also
    needs float64; call :func:`downcast_mne_data` when finished.
    """
    filtered = data.copy()
    upcast_mne_data(filtered, dtype=_FILTER_DTYPE)
    filtered.filter(*args, **kwargs)
    if restore_working_dtype:
        downcast_mne_data(filtered)
    return filtered


def filter_array(data: Any, sfreq: float, *, restore_working_dtype: bool = True, **kwargs):
    """Run ``mne.filter.filter_data`` with a float64 upcast (and optional downcast)."""
    import mne

    arr64 = np.asarray(data, dtype=_FILTER_DTYPE)
    filtered = mne.filter.filter_data(arr64, sfreq, **kwargs)
    if restore_working_dtype:
        return np.asarray(filtered, dtype=_WORKING_DTYPE)
    return np.asarray(filtered, dtype=_FILTER_DTYPE)
