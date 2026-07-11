"""Tests for beginner-facing builders and legacy compatibility shims."""

from __future__ import annotations

import warnings

import pytest

from LFPAnalysis import (
    build_basic_pipeline_config,
    build_event_locked_pipeline_config,
    build_spectral_pipeline_config,
)
from LFPAnalysis.exceptions import ConfigurationError
from LFPAnalysis import legacy


@pytest.mark.unit
def test_build_basic_pipeline_config_returns_stable_defaults():
    config = build_basic_pipeline_config("data/sample_ieeg.fif", file_format="mne")
    assert config.load.file_format == "mne"
    assert config.reference.method == "none"
    assert config.epoch.enabled is False
    assert config.spectral.enabled is False


@pytest.mark.unit
def test_build_event_locked_pipeline_config_enables_epoch_and_baseline():
    config = build_event_locked_pipeline_config(
        "data/sample_ieeg_continuous_rest.fif",
        event_name="demo",
        event_times=[1.0, 2.0],
    )
    assert config.epoch.enabled is True
    assert config.baseline.enabled is True
    assert config.epoch.event_name == "demo"


@pytest.mark.unit
def test_build_spectral_pipeline_config_rejects_nonstable_methods():
    with pytest.raises(ConfigurationError):
        build_spectral_pipeline_config("data/sample_ieeg.fif", spectral_method="connectivity")


@pytest.mark.integration
def test_legacy_make_mne_warns_and_loads_stable_mne_path(tmp_path, synthetic_raw):
    path = tmp_path / "synthetic-raw.fif"
    synthetic_raw.save(path, overwrite=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Legacy default resample_sr=500; pass the source rate to avoid resampling.
        loaded = legacy.make_mne(load_path=path, format="mne", resample_sr=int(synthetic_raw.info["sfreq"]))
    assert loaded.info["sfreq"] == synthetic_raw.info["sfreq"]
    assert loaded._data.dtype.name == "float32"
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)


@pytest.mark.integration
def test_legacy_ref_mne_warns_and_returns_identity_for_none(synthetic_raw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reref = legacy.ref_mne(mne_data=synthetic_raw, elec_path=None, method="none")
    # method='none' intentionally avoids a defensive copy to save RAM.
    assert reref is synthetic_raw
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)


@pytest.mark.integration
def test_legacy_compute_and_baseline_tfr_warns(monkeypatch):
    called = {}

    class DummyLegacy:
        @staticmethod
        def compute_and_baseline_tfr(*args, **kwargs):
            called["ok"] = True
            return "tfr"

    monkeypatch.setattr(legacy, "_legacy_preprocess_module", lambda: DummyLegacy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = legacy.compute_and_baseline_tfr(1, 2)
    assert out == "tfr"
    assert called["ok"] is True
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)


@pytest.mark.integration
def test_legacy_make_mne_requires_path_for_mne_format():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(ConfigurationError):
            legacy.make_mne(load_path=None, format="mne")


@pytest.mark.integration
def test_legacy_make_epochs_warns_and_uses_stable_path(tmp_path, synthetic_raw):
    path = tmp_path / "synthetic-raw.fif"
    synthetic_raw.save(path, overwrite=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        epochs = legacy.make_epochs(
            load_path=path,
            behav_name="demo",
            behav_times=[5.0, 10.0, 15.0],
            ev_start_s=0.5,
            ev_end_s=1.0,
        )
    assert len(epochs) == 3
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)


@pytest.mark.integration
def test_legacy_make_epochs_requires_args():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(ConfigurationError):
            legacy.make_epochs(load_path=None, behav_name=None, behav_times=None)

