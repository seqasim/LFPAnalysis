"""Unit tests for workflow-layer behavior that does not require full analysis extras."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from LFPAnalysis import ArtifactConfig, BaselineConfig, detect_artifacts
from LFPAnalysis.config import EpochConfig, LoadConfig, ReferenceConfig
from LFPAnalysis.exceptions import ConfigurationError
from LFPAnalysis.schemas import ARTIFACT_EVENT_COLUMNS
from LFPAnalysis.workflow import (
    _downcast_mne_data,
    _get_data_array,
    baseline_lfp,
    compute_tfr_features,
    derive_referenced_electrode_df,
    load_lfp,
    load_electrode_metadata,
    make_epochs,
    preprocess_lfp,
)


@pytest.mark.unit
def test_detect_artifacts_accepts_custom_mapping():
    dummy = SimpleNamespace(info={"sfreq": 100.0})
    result = detect_artifacts(
        dummy,
        ArtifactConfig(methods=["custom"], custom_detector=lambda data: {"l1": [0.1, 0.3]}),
    )
    assert list(result["custom"].columns) == list(ARTIFACT_EVENT_COLUMNS)
    assert len(result["custom"]) == 2


@pytest.mark.unit
def test_detect_artifacts_accepts_custom_dataframe():
    dummy = SimpleNamespace(info={"sfreq": 100.0})
    dataframe = pd.DataFrame({"event_kind": ["custom"], "channel": ["l1"], "time_seconds": [0.1], "sample_index": [10]})
    result = detect_artifacts(
        dummy,
        ArtifactConfig(methods=["custom"], custom_detector=lambda data: dataframe),
    )
    assert result["custom"].equals(dataframe)


@pytest.mark.unit
def test_baseline_lfp_noop_when_disabled():
    data = object()
    returned, summary = baseline_lfp(data, BaselineConfig(enabled=False))
    assert returned is data
    assert summary.empty


@pytest.mark.unit
def test_detect_artifacts_custom_invalid_return_type():
    dummy = SimpleNamespace(info={"sfreq": 100.0})
    with pytest.raises(ConfigurationError):
        detect_artifacts(
            dummy,
            ArtifactConfig(methods=["custom"], custom_detector=lambda data: ["not", "valid"]),
        )


@pytest.mark.unit
def test_preprocess_requires_electrode_path_for_reference():
    dummy = SimpleNamespace(copy=lambda: dummy)
    with pytest.raises(ConfigurationError):
        preprocess_lfp(dummy, ReferenceConfig(method="bipolar", electrode_path=None))


@pytest.mark.unit
def test_preprocess_rejects_non_copyable_for_reference():
    with pytest.raises(ConfigurationError):
        preprocess_lfp(object(), ReferenceConfig(method="bipolar", electrode_path="x.csv"))


@pytest.mark.unit
def test_load_electrode_metadata_xlsx(tmp_path):
    path = tmp_path / "electrodes.xlsx"
    pd.DataFrame(
        {
            "label": ["l1"],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        }
    ).to_excel(path, index=False)
    dataframe = load_electrode_metadata(path)
    assert list(dataframe["label"]) == ["l1"]


@pytest.mark.unit
def test_load_electrode_metadata_accepts_nmmlabel(tmp_path):
    path = tmp_path / "electrodes.xlsx"
    pd.DataFrame({"NMMlabel": ["l1"], "x": [1.0]}).to_excel(path, index=False)
    dataframe = load_electrode_metadata(path)
    assert list(dataframe["label"]) == ["l1"]


@pytest.mark.unit
def test_load_electrode_metadata_rejects_bad_suffix(tmp_path):
    path = tmp_path / "electrodes.txt"
    path.write_text("nope")
    with pytest.raises(ConfigurationError):
        load_electrode_metadata(path)


@pytest.mark.unit
def test_get_data_array_fallback_without_private_buffer():
    class FakeRaw:
        preload = True
        _data = None

        def get_data(self, copy=False):
            return np.ones((2, 5), dtype=np.float64)

    arr = _get_data_array(FakeRaw(), copy=False)
    assert arr.dtype == np.dtype("float32")
    assert arr.shape == (2, 5)


@pytest.mark.unit
def test_get_data_array_typeerror_fallback():
    class FakeRaw:
        preload = True
        _data = None

        def get_data(self):
            return np.ones((1, 4), dtype=np.float64)

    arr = _get_data_array(FakeRaw(), copy=True)
    assert arr.dtype == np.dtype("float32")


@pytest.mark.unit
def test_downcast_restores_dtype_even_when_preload_false():
    class FakeRaw:
        preload = False
        _data = np.ones((1, 3), dtype=np.float64)

    out = _downcast_mne_data(FakeRaw())
    assert out._data.dtype == np.dtype("float32")


@pytest.mark.unit
def test_downcast_skips_when_data_missing():
    class FakeRaw:
        preload = False
        _data = None

    out = _downcast_mne_data(FakeRaw())
    assert out._data is None


@pytest.mark.unit
def test_make_epochs_requires_event_times():
    dummy = SimpleNamespace(info={"sfreq": 100.0}, copy=lambda: dummy)
    with pytest.raises(ConfigurationError):
        make_epochs(dummy, EpochConfig(enabled=True, event_times=[]))


@pytest.mark.unit
def test_make_epochs_requires_raw_like():
    with pytest.raises(ConfigurationError):
        make_epochs(object(), EpochConfig(enabled=True, event_times=[1.0]))


@pytest.mark.unit
def test_baseline_requires_get_data_for_raw_like():
    with pytest.raises(ConfigurationError):
        baseline_lfp(SimpleNamespace(), BaselineConfig(enabled=True, mode="zscore", baseline_window=(0, 1)))


@pytest.mark.unit
def test_derive_referenced_electrode_df_bipolar_midpoint():
    elec = pd.DataFrame(
        {
            "label": ["a1", "a2"],
            "mni_x": [0.0, 10.0],
            "mni_y": [2.0, 6.0],
            "mni_z": [4.0, 8.0],
            "roi": ["ACC", "ACC"],
        }
    )
    derived = derive_referenced_electrode_df(elec, ["a1-a2"], method="bipolar")
    assert list(derived["label"]) == ["a1-a2"]
    assert derived.loc[0, "anode"] == "a1"
    assert derived.loc[0, "cathode"] == "a2"
    assert derived.loc[0, "mni_x"] == pytest.approx(5.0)
    assert derived.loc[0, "mni_y"] == pytest.approx(4.0)
    assert derived.loc[0, "mni_z"] == pytest.approx(6.0)
    assert derived.loc[0, "roi"] == "ACC"


@pytest.mark.unit
def test_derive_referenced_electrode_df_wm_inherits_anode():
    elec = pd.DataFrame({"label": ["g1", "w1"], "mni_x": [1.5, 99.0], "roi": ["GM", "WM"]})
    derived = derive_referenced_electrode_df(elec, ["g1-w1"], method="wm")
    assert derived.loc[0, "mni_x"] == pytest.approx(1.5)
    assert derived.loc[0, "roi"] == "GM"
    assert derived.loc[0, "anode"] == "g1"
    assert derived.loc[0, "cathode"] == "w1"


@pytest.mark.unit
def test_derive_referenced_electrode_df_passthrough_for_name_preserving_methods():
    elec = pd.DataFrame({"label": ["a1", "a2"], "mni_x": [1.0, 2.0]})
    for method in ("none", "car", "car_trimmed"):
        derived = derive_referenced_electrode_df(elec, ["a1", "a2"], method=method)
        assert derived.equals(elec)


@pytest.mark.unit
def test_derive_referenced_electrode_df_warns_on_missing_contact():
    elec = pd.DataFrame({"label": ["a1"], "mni_x": [1.0]})
    with pytest.warns(RuntimeWarning, match="Could not derive electrode row"):
        derived = derive_referenced_electrode_df(elec, ["a1-a2"], method="bipolar")
    assert derived.empty


@pytest.mark.unit
def test_make_epochs_drops_nan_and_none_times_with_metadata(mne_module):
    sfreq = 100.0
    raw = mne_module.io.RawArray(
        np.vstack([np.sin(np.arange(5000) / sfreq)]),
        mne_module.create_info(["l1"], sfreq, ch_types=["seeg"]),
        verbose=False,
    )
    config = EpochConfig(
        enabled=True,
        event_name="ev",
        event_times=[1.0, np.nan, 2.0, "None"],
        tmin=-0.2,
        tmax=0.3,
        metadata={"trial": [10, 20, 30, 40]},
    )
    epochs = make_epochs(raw, config)
    assert len(epochs) == 2
    assert list(epochs.metadata["trial"]) == [10, 30]


@pytest.mark.unit
def test_compute_tfr_features_cross_event_uses_trialwise_baseline(monkeypatch, mne_module):
    sfreq = 50.0
    times = np.arange(0, 20, 1 / sfreq)
    raw = mne_module.io.RawArray(
        np.vstack([np.sin(2 * np.pi * 3 * times)]),
        mne_module.create_info(["l1"], sfreq, ch_types=["seeg"]),
        verbose=False,
    )
    events = np.array([[200, 0, 1], [500, 0, 1]])
    task_epochs = mne_module.Epochs(
        raw.copy(),
        events=events,
        event_id={"task": 1},
        tmin=-0.5,
        tmax=0.5,
        baseline=None,
        preload=True,
        verbose=False,
    )
    baseline_events = events.copy()
    baseline_events[:, 0] = baseline_events[:, 0] + 20
    baseline_epochs = mne_module.Epochs(
        raw.copy(),
        events=baseline_events,
        event_id={"baseline": 1},
        tmin=-0.5,
        tmax=0.5,
        baseline=None,
        preload=True,
        verbose=False,
    )

    from LFPAnalysis.config import TfrConfig
    import LFPAnalysis.workflow as workflow_mod

    calls = {"count": 0}

    def _fake_trialwise(**kwargs):
        calls["count"] += 1
        return kwargs["data"] - np.nanmean(kwargs["baseline_mne"], axis=(0, 3), keepdims=True)

    monkeypatch.setattr(
        workflow_mod,
        "_legacy_preprocess_module",
        lambda: SimpleNamespace(baseline_trialwise_TFR=_fake_trialwise, _nan_mask_tfr_events=lambda *a, **k: None),
    )

    tfr = compute_tfr_features(
        task_epochs,
        TfrConfig(enabled=True, method="morlet", freqs=[4, 8], n_cycles=2.0, crop_tmin=-0.2, crop_tmax=0.2),
        baseline_epochs=baseline_epochs,
        artifact_tables={},
    )
    assert tfr["method"] == "morlet"
    assert calls["count"] >= 1
    assert tfr["metadata"]["baseline_mode"].iloc[0] == "trialwise"


@pytest.mark.unit
def test_load_lfp_applies_clinical_default_notch_and_resample(monkeypatch):
    class FakeRaw:
        def __init__(self):
            self.ch_names = ["a1"]
            self.info = {"line_freq": None, "bads": []}
            self._data = np.ones((1, 10), dtype=np.float32)
            self.preload = True
            self.calls = []

        def get_channel_types(self):
            return ["seeg"]

        def notch_filter(self, freqs):
            self.calls.append(("notch", tuple(freqs)))

        def resample(self, sfreq):
            self.calls.append(("resample", float(sfreq)))

    import LFPAnalysis.workflow as workflow_mod

    fake_raw = FakeRaw()
    fake_mne = SimpleNamespace(io=SimpleNamespace(read_raw_edf=lambda *_args, **_kwargs: fake_raw))
    monkeypatch.setattr(workflow_mod, "_require_mne", lambda: fake_mne)
    monkeypatch.setattr(workflow_mod, "resolve_existing_path", lambda path, field_name=None: Path(path))

    loaded = load_lfp(LoadConfig(path="demo.edf", file_format="edf", preload=True))
    assert loaded is fake_raw
    assert ("notch", (60.0, 120.0, 180.0, 240.0)) in fake_raw.calls
    assert ("resample", 500.0) in fake_raw.calls


@pytest.mark.unit
def test_load_lfp_skips_clinical_resample_when_explicit_none(monkeypatch):
    class FakeRaw:
        def __init__(self):
            self.ch_names = ["a1"]
            self.info = {"line_freq": None, "bads": []}
            self._data = np.ones((1, 10), dtype=np.float32)
            self.preload = True
            self.calls = []

        def get_channel_types(self):
            return ["seeg"]

        def notch_filter(self, freqs):
            self.calls.append(("notch", tuple(freqs)))

        def resample(self, sfreq):
            self.calls.append(("resample", float(sfreq)))

    import LFPAnalysis.workflow as workflow_mod

    fake_raw = FakeRaw()
    fake_mne = SimpleNamespace(io=SimpleNamespace(read_raw_edf=lambda *_args, **_kwargs: fake_raw))
    monkeypatch.setattr(workflow_mod, "_require_mne", lambda: fake_mne)
    monkeypatch.setattr(workflow_mod, "resolve_existing_path", lambda path, field_name=None: Path(path))

    loaded = load_lfp(
        LoadConfig(
            path="demo.edf",
            file_format="edf",
            preload=True,
            resample_sfreq=None,
            notch_freqs=None,
        )
    )
    assert loaded is fake_raw
    assert fake_raw.calls == []
