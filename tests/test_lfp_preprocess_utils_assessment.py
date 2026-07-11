import importlib
import sys
import types

import numpy as np


def _import_lfp_preprocess_utils(monkeypatch):
    sys.modules.pop("LFPAnalysis.lfp_preprocess_utils", None)

    mne_stub = types.ModuleType("mne")
    preprocessing_stub = types.ModuleType("mne.preprocessing")
    bads_stub = types.ModuleType("mne.preprocessing.bads")
    bads_stub._find_outliers = lambda *args, **kwargs: None
    filter_stub = types.ModuleType("mne.filter")
    filter_stub.next_fast_len = lambda value: value

    epochs_type = type("Epochs", (), {})
    epochs_tfr_type = type("EpochsTFR", (), {})
    mne_stub.preprocessing = preprocessing_stub
    preprocessing_stub.bads = bads_stub
    mne_stub.filter = filter_stub
    mne_stub.epochs = types.SimpleNamespace(Epochs=epochs_type)
    mne_stub.time_frequency = types.SimpleNamespace(
        tfr=types.SimpleNamespace(EpochsTFR=epochs_tfr_type)
    )

    levenshtein_stub = types.ModuleType("Levenshtein")
    levenshtein_stub.distance = lambda left, right: abs(len(left) - len(right))

    monkeypatch.setitem(sys.modules, "mne", mne_stub)
    monkeypatch.setitem(sys.modules, "mne.preprocessing", preprocessing_stub)
    monkeypatch.setitem(sys.modules, "mne.preprocessing.bads", bads_stub)
    monkeypatch.setitem(sys.modules, "mne.filter", filter_stub)
    monkeypatch.setitem(sys.modules, "Levenshtein", levenshtein_stub)

    return importlib.import_module("LFPAnalysis.lfp_preprocess_utils")


def test_mean_baseline_time_supports_2d_inputs(monkeypatch):
    lfp_preprocess_utils = _import_lfp_preprocess_utils(monkeypatch)

    data = np.array([[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    baseline = np.array([[1.0, 3.0], [2.0, 4.0]])

    result = lfp_preprocess_utils.mean_baseline_time(data, baseline, mode="zscore")
    expected = (data - baseline.mean(axis=-1, keepdims=True)) / baseline.std(
        axis=-1, keepdims=True
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_baseline_trialwise_tfr_uses_broadcasting_equivalent_math(monkeypatch):
    lfp_preprocess_utils = _import_lfp_preprocess_utils(monkeypatch)

    data = np.arange(1, 49, dtype=float).reshape(2, 2, 2, 6)
    baseline = np.arange(1, 33, dtype=float).reshape(2, 2, 2, 4)

    result = lfp_preprocess_utils.baseline_trialwise_TFR(
        data=data,
        baseline_mne=baseline,
        mode="mean",
        include_epoch_in_baseline=False,
    )

    reshaped_baseline = baseline.transpose(1, 2, 3, 0).reshape(2, 2, -1)
    baseline_mean = np.nanmean(reshaped_baseline, axis=-1).reshape(1, 2, 2, 1)
    expected = data - baseline_mean

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_baseline_tfr_permute_matches_manual_sampling(monkeypatch):
    lfp_preprocess_utils = _import_lfp_preprocess_utils(monkeypatch)

    data = np.array(
        [
            [
                [[2.0, 4.0], [6.0, 8.0]],
                [[10.0, 12.0], [14.0, 16.0]],
            ]
        ]
    )
    baseline = np.array(
        [
            [
                [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
                [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]],
            ]
        ]
    )
    sample_indices = np.array(
        [
            [[0, 1, 2], [2, 1, 0]],
            [[1, 1, 2], [0, 2, 2]],
        ]
    )

    monkeypatch.setattr(
        lfp_preprocess_utils.np.random,
        "randint",
        lambda low, high, size: sample_indices,
    )

    result = lfp_preprocess_utils.baseline_TFR_permute(
        data=data,
        baseline_mne=baseline,
        mode="zscore",
        num_samples=3,
    )

    baseline_flat = np.moveaxis(baseline, (1, 2), (0, 1)).reshape(2, 2, -1)
    sampled = np.take_along_axis(baseline_flat, sample_indices, axis=-1)
    mean = np.nanmean(sampled, axis=-1).reshape(1, 2, 2, 1)
    std = np.nanstd(sampled, axis=-1).reshape(1, 2, 2, 1)
    expected = (data - mean) / std

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
