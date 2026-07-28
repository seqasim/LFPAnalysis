"""Unit tests for workflow-layer behavior that does not require full analysis extras."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from LFPAnalysis import ArtifactConfig, BaselineConfig, detect_artifacts
from LFPAnalysis.config import EpochConfig, ReferenceConfig
from LFPAnalysis.exceptions import ConfigurationError
from LFPAnalysis.schemas import ARTIFACT_EVENT_COLUMNS
from LFPAnalysis.workflow import (
    _downcast_mne_data,
    _get_data_array,
    baseline_lfp,
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
