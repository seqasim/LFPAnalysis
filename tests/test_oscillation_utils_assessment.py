import importlib
import sys
import types

import numpy as np


def _import_oscillation_utils(monkeypatch):
    sys.modules.pop("LFPAnalysis.oscillation_utils", None)

    mne_stub = types.ModuleType("mne")
    mne_stub.Epochs = type("Epochs", (), {})
    mne_stub.EpochsArray = type("EpochsArray", (), {})
    mne_stub.pick_info = lambda info, sel, copy=True: info

    time_frequency_stub = types.ModuleType("mne.time_frequency")
    time_frequency_stub.EpochsTFR = type("EpochsTFR", (), {})
    time_frequency_stub.EpochsTFRArray = type("EpochsTFRArray", (), {})

    filter_stub = types.ModuleType("mne.filter")
    filter_stub.next_fast_len = lambda value: value

    mne_connectivity_stub = types.ModuleType("mne_connectivity")
    mne_connectivity_stub.phase_slope_index = lambda *args, **kwargs: None
    mne_connectivity_stub.spectral_connectivity_epochs = lambda *args, **kwargs: None
    mne_connectivity_stub.spectral_connectivity_time = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "mne", mne_stub)
    monkeypatch.setitem(sys.modules, "mne.time_frequency", time_frequency_stub)
    monkeypatch.setitem(sys.modules, "mne.filter", filter_stub)
    monkeypatch.setitem(sys.modules, "mne_connectivity", mne_connectivity_stub)

    return importlib.import_module("LFPAnalysis.oscillation_utils")


def test_swap_time_blocks_batch_rotates_each_series(monkeypatch):
    oscillation_utils = _import_oscillation_utils(monkeypatch)

    data = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
    cutpoints = np.array([1, 3])

    result = oscillation_utils._swap_time_blocks_batch(data, cutpoints)
    expected = np.array([[2, 3, 4, 1], [40, 10, 20, 30]])

    np.testing.assert_array_equal(result, expected)


def test_make_surrogate_arrays_swap_time_blocks_is_deterministic(monkeypatch):
    oscillation_utils = _import_oscillation_utils(monkeypatch)

    data = np.arange(12, dtype=float).reshape(3, 4)
    first = oscillation_utils.make_surrogate_arrays(
        data,
        method="swap_time_blocks",
        n_shuffles=2,
        rng_seed=5,
        return_generator=False,
    )
    second = oscillation_utils.make_surrogate_arrays(
        data,
        method="swap_time_blocks",
        n_shuffles=2,
        rng_seed=5,
        return_generator=False,
    )

    expected_cutpoints = np.random.default_rng(5).integers(1, data.shape[1], data.shape[0])
    expected_first = oscillation_utils._swap_time_blocks_batch(data, expected_cutpoints)

    np.testing.assert_array_equal(first[0], expected_first)
    np.testing.assert_array_equal(first[0], second[0])
