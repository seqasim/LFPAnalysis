"""Cross-module integration checks for the advanced utility layer."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from LFPAnalysis import statistics_utils, sync_utils


def _import_utility_stack(monkeypatch):
    """Import utility modules with lightweight dependency stubs."""

    for name in [
        "LFPAnalysis.analysis_utils",
        "LFPAnalysis.iowa_utils",
        "LFPAnalysis.lfp_preprocess_utils",
        "LFPAnalysis.nlx_utils",
        "LFPAnalysis.oscillation_utils",
    ]:
        sys.modules.pop(name, None)

    mne_stub = types.ModuleType("mne")
    preprocessing_stub = types.ModuleType("mne.preprocessing")
    bads_stub = types.ModuleType("mne.preprocessing.bads")
    bads_stub._find_outliers = lambda *args, **kwargs: None
    filter_stub = types.ModuleType("mne.filter")
    filter_stub.next_fast_len = lambda value: value
    time_frequency_stub = types.ModuleType("mne.time_frequency")
    mne_stub.preprocessing = preprocessing_stub
    preprocessing_stub.bads = bads_stub
    mne_stub.filter = filter_stub
    mne_stub.Epochs = type("Epochs", (), {})
    mne_stub.EpochsArray = type("EpochsArray", (), {})
    mne_stub.pick_info = lambda info, sel, copy=True: info
    mne_stub.epochs = types.SimpleNamespace(Epochs=type("Epochs", (), {}))
    epochs_tfr_type = type("EpochsTFR", (), {})
    epochs_tfr_array_type = type("EpochsTFRArray", (), {})
    time_frequency_stub.tfr = types.SimpleNamespace(EpochsTFR=epochs_tfr_type)
    time_frequency_stub.EpochsTFR = epochs_tfr_type
    time_frequency_stub.EpochsTFRArray = epochs_tfr_array_type
    mne_stub.time_frequency = time_frequency_stub

    fooof_stub = types.ModuleType("fooof")
    fooof_stub.FOOOFGroup = type("DummyFOOOFGroup", (), {})
    fooof_stub.analysis = types.SimpleNamespace(get_band_peak_fm=lambda *args, **kwargs: None)
    pycatch22_stub = types.ModuleType("pycatch22")
    levenshtein_stub = types.ModuleType("Levenshtein")
    levenshtein_stub.distance = lambda left, right: abs(len(left) - len(right))
    mne_connectivity_stub = types.ModuleType("mne_connectivity")
    mne_connectivity_stub.phase_slope_index = lambda *args, **kwargs: None
    mne_connectivity_stub.spectral_connectivity_epochs = lambda *args, **kwargs: None
    mne_connectivity_stub.spectral_connectivity_time = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "mne", mne_stub)
    monkeypatch.setitem(sys.modules, "mne.preprocessing", preprocessing_stub)
    monkeypatch.setitem(sys.modules, "mne.preprocessing.bads", bads_stub)
    monkeypatch.setitem(sys.modules, "mne.filter", filter_stub)
    monkeypatch.setitem(sys.modules, "mne.time_frequency", time_frequency_stub)
    monkeypatch.setitem(sys.modules, "fooof", fooof_stub)
    monkeypatch.setitem(sys.modules, "pycatch22", pycatch22_stub)
    monkeypatch.setitem(sys.modules, "Levenshtein", levenshtein_stub)
    monkeypatch.setitem(sys.modules, "mne_connectivity", mne_connectivity_stub)

    return {
        "analysis_utils": importlib.import_module("LFPAnalysis.analysis_utils"),
        "iowa_utils": importlib.import_module("LFPAnalysis.iowa_utils"),
        "lfp_preprocess_utils": importlib.import_module("LFPAnalysis.lfp_preprocess_utils"),
        "nlx_utils": importlib.import_module("LFPAnalysis.nlx_utils"),
        "oscillation_utils": importlib.import_module("LFPAnalysis.oscillation_utils"),
    }


@pytest.mark.unit
def test_iowa_and_neuralynx_utils_share_channel_names(tmp_path: Path, monkeypatch):
    modules = _import_utility_stack(monkeypatch)
    iowa_utils = modules["iowa_utils"]
    nlx_utils = modules["nlx_utils"]

    connect_table = pd.DataFrame(
        {
            "Code": ["scalp", "CAN", "EKG", "UNUSED", "DepthA"],
            "Contact Location": ["Scalp A", "Resp belt", "EKG", "Unused", "Left Hipp"],
            "NLX-LFPx channel": ["1", "2", "3", "4", "5:6"],
        }
    )
    connect_path = tmp_path / "connect_table.csv"
    connect_table.to_csv(connect_path, index=False)

    eeg_names, resp_names, ekg_names, seeg_names, drop_names = iowa_utils.extract_names_connect_table(
        str(connect_path)
    )

    monkeypatch.setattr(
        nlx_utils,
        "load_ncs",
        lambda path: {"data": np.arange(10, dtype=float), "sampling_rate": 1000},
    )
    signals, srs, ch_names, ch_types = nlx_utils.parse_subject_nlx_data(
        [
            "subject/LFPx1.ncs",
            "subject/LFPx2.ncs",
            "subject/LFPx3.ncs",
            "subject/LFPx4.ncs",
            "subject/LFPx5.ncs",
            "subject/LFPx6.ncs",
        ],
        eeg_names=eeg_names,
        resp_names=resp_names,
        ekg_names=ekg_names,
        seeg_names=seeg_names,
        drop_names=drop_names,
    )

    assert len(signals) == 5
    assert srs == [1000, 1000, 1000, 1000, 1000]
    assert ch_names == ["lfpx1", "lfpx2", "lfpx3", "lfpx5", "lfpx6"]
    assert ch_types == ["eeg", "bio", "ecg", "seeg", "seeg"]


@pytest.mark.unit
def test_sync_output_can_drive_analysis_helpers(monkeypatch):
    beh_ts = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
    neural_ts = beh_ts + 0.25

    good_beh, good_neural = sync_utils.pulsealign(beh_ts, neural_ts, windSize=3)
    assert len(good_beh) >= 3

    event_times = np.array([1.0, 2.0, 3.5, 5.0])
    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 400))

    # Import only the analysis helper with light stubs.
    sys.modules.pop("LFPAnalysis.analysis_utils", None)
    mne_stub = types.ModuleType("mne")
    fooof_stub = types.ModuleType("fooof")
    fooof_stub.FOOOFGroup = type("DummyFOOOFGroup", (), {})
    pycatch22_stub = types.ModuleType("pycatch22")
    monkeypatch.setitem(sys.modules, "mne", mne_stub)
    monkeypatch.setitem(sys.modules, "fooof", fooof_stub)
    monkeypatch.setitem(sys.modules, "pycatch22", pycatch22_stub)
    analysis_utils = importlib.import_module("LFPAnalysis.analysis_utils")

    sta, ste = analysis_utils.lfp_sta(
        ev_times=event_times + (good_neural[0] - good_beh[0]),
        signal=signal,
        sr=20.0,
        pre=0.25,
        post=0.25,
    )

    assert sta.shape == ste.shape
    assert sta.ndim == 1
    assert np.isfinite(sta).all()


@pytest.mark.unit
def test_preprocess_surrogates_and_statistics_share_shapes(monkeypatch):
    modules = _import_utility_stack(monkeypatch)
    lfp_preprocess_utils = modules["lfp_preprocess_utils"]
    oscillation_utils = modules["oscillation_utils"]

    data = np.abs(np.arange(1, 49, dtype=float).reshape(1, 6, 8)) + 0.5
    baseline = np.abs(np.arange(1, 25, dtype=float).reshape(1, 6, 4)) + 0.5

    baselined = lfp_preprocess_utils.baseline_avg_TFR(data, baseline, mode="zscore")
    surrogate_arrays = oscillation_utils.make_surrogate_arrays(
        baselined[0],
        method="swap_time_blocks",
        n_shuffles=2,
        rng_seed=7,
        return_generator=False,
    )

    surrogate_summary = surrogate_arrays[0].mean(axis=1)
    permutation_outputs = iter(
        [
            surrogate_summary[::-1],
            np.roll(surrogate_summary, 1),
        ]
    )
    monkeypatch.setattr(
        statistics_utils.np.random,
        "permutation",
        lambda values: next(permutation_outputs),
    )

    regression_df = pd.DataFrame(
        {
            "y": surrogate_summary,
            "x": np.linspace(-1.0, 1.0, len(surrogate_summary)),
        }
    )
    results = statistics_utils.permutation_regression_zscore(
        regression_df,
        "y ~ x",
        n_permutations=2,
    )

    assert set(results["predictor"]) == {"Intercept", "x"}
    assert np.isfinite(results.loc[results["predictor"] == "x", "z_beta"]).all()
