import importlib
import sys
import types

import numpy as np
import pandas as pd


def _import_analysis_utils(monkeypatch):
    sys.modules.pop("LFPAnalysis.analysis_utils", None)

    mne_stub = types.ModuleType("mne")
    fooof_stub = types.ModuleType("fooof")
    fooof_stub.FOOOFGroup = type("DummyFOOOFGroup", (), {})
    pycatch22_stub = types.ModuleType("pycatch22")

    monkeypatch.setitem(sys.modules, "mne", mne_stub)
    monkeypatch.setitem(sys.modules, "fooof", fooof_stub)
    monkeypatch.setitem(sys.modules, "pycatch22", pycatch22_stub)

    return importlib.import_module("LFPAnalysis.analysis_utils")


def test_rolling_rms_last_axis_matches_trailing_window(monkeypatch):
    analysis_utils = _import_analysis_utils(monkeypatch)

    data = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    result = analysis_utils._rolling_rms_last_axis(data, window_samples=2)

    expected = np.array(
        [[[
            1.0,
            np.sqrt((1.0**2 + 2.0**2) / 2.0),
            np.sqrt((2.0**2 + 3.0**2) / 2.0),
            np.sqrt((3.0**2 + 4.0**2) / 2.0),
        ]]]
    )
    np.testing.assert_allclose(result, expected)


def test_select_rois_picks_caches_yba_lookup(monkeypatch):
    analysis_utils = _import_analysis_utils(monkeypatch)
    analysis_utils._load_yba_roi_labels.cache_clear()

    read_calls = []

    def fake_read_excel(path):
        read_calls.append(path)
        return pd.DataFrame(
            {
                "Long.name": ["hippocampus"],
                "Custom": ["HPC"],
            }
        )

    monkeypatch.setattr(
        analysis_utils.pkg_resources,
        "resource_filename",
        lambda package, resource: "/tmp",
    )
    monkeypatch.setattr(analysis_utils.pd, "read_excel", fake_read_excel)

    elec_data = pd.DataFrame(
        {
            "label": ["chan1"],
            "NMM": ["hippocampus"],
            "BN246": ["hippocampus"],
            "YBA_1": ["hippocampus"],
            "collapsed_manual": [np.nan],
        }
    )

    assert analysis_utils.select_rois_picks(elec_data, "chan1") == "HPC"
    assert analysis_utils.select_rois_picks(elec_data, "chan1") == "HPC"
    assert len(read_calls) == 1
