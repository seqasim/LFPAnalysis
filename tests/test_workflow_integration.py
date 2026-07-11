"""Integration tests for the stable workflow API using synthetic MNE objects."""

from __future__ import annotations

import numpy as np
import pytest

from LFPAnalysis import (
    ArtifactConfig,
    BaselineConfig,
    EpochConfig,
    LoadConfig,
    PipelineConfig,
    ReferenceConfig,
    SpectralConfig,
    baseline_lfp,
    build_basic_pipeline_config,
    build_spectral_pipeline_config,
    compute_spectral_features,
    detect_artifacts,
    load_lfp,
    make_epochs,
    preprocess_lfp,
    run_pipeline,
)
from LFPAnalysis.exceptions import ConfigurationError
from LFPAnalysis.workflow import (
    _downcast_mne_data,
    _ensure_preloaded,
    _get_data_array,
)


@pytest.mark.integration
def test_load_lfp_accepts_raw_object(synthetic_raw):
    loaded = load_lfp(LoadConfig(path=synthetic_raw, file_format="mne"))
    assert loaded.info["sfreq"] == synthetic_raw.info["sfreq"]
    assert loaded.ch_names == synthetic_raw.ch_names


@pytest.mark.integration
def test_make_epochs_creates_expected_number_of_trials(synthetic_raw):
    epochs = make_epochs(
        synthetic_raw,
        EpochConfig(enabled=True, event_name="demo", event_times=[5.0, 10.0, 15.0], tmin=-0.5, tmax=1.0),
    )
    assert len(epochs) == 3
    assert epochs.event_id == {"demo": 1}


@pytest.mark.integration
def test_baseline_lfp_epochs_returns_summary(synthetic_epochs):
    baselined, summary = baseline_lfp(
        synthetic_epochs,
        BaselineConfig(enabled=True, mode="zscore", baseline_window=(-0.5, 0.0)),
    )
    assert baselined.get_data().shape == synthetic_epochs.get_data().shape
    assert set(summary["channel"]) == set(synthetic_epochs.ch_names)


@pytest.mark.integration
def test_run_pipeline_returns_structured_outputs(synthetic_raw):
    config = PipelineConfig(
        load=LoadConfig(path=synthetic_raw, file_format="mne"),
        reference=ReferenceConfig(method="none"),
        artifact=ArtifactConfig(methods=["custom"], custom_detector=lambda data: {"l1": [0.1]}),
        baseline=BaselineConfig(enabled=False, mode="none"),
        epoch=EpochConfig(enabled=True, event_name="demo", event_times=[5.0, 10.0, 15.0], tmin=-0.5, tmax=1.0),
        spectral=SpectralConfig(enabled=False, method="none"),
    )
    result = run_pipeline(config)
    # Continuous stages are freed when epoching is enabled to reduce peak RAM.
    assert result.raw is None
    assert result.referenced is None
    assert result.epochs is not None
    assert result.epochs._data.dtype == np.dtype("float32")
    assert "custom" in result.artifact_tables
    assert result.metadata["reference_method"] == "none"
    assert result.metadata["working_dtype"] == "float32"


@pytest.mark.integration
def test_load_lfp_downcasts_to_float32(synthetic_raw):
    loaded = load_lfp(LoadConfig(path=synthetic_raw, file_format="mne", preload=True))
    assert loaded.preload
    assert loaded._data.dtype == np.dtype("float32")


@pytest.mark.integration
def test_run_pipeline_keeps_continuous_when_not_epoching(synthetic_raw):
    config = PipelineConfig(
        load=LoadConfig(path=synthetic_raw, file_format="mne"),
        reference=ReferenceConfig(method="none"),
        artifact=ArtifactConfig(methods=["none"]),
        baseline=BaselineConfig(enabled=False, mode="none"),
        epoch=EpochConfig(enabled=False),
        spectral=SpectralConfig(enabled=False, method="none"),
    )
    result = run_pipeline(config)
    assert result.raw is not None
    assert result.referenced is result.raw
    assert result.epochs is None
    assert result.raw._data.dtype == np.dtype("float32")


@pytest.mark.integration
def test_load_lfp_from_fif_lazy_and_pick_channels(tmp_path, synthetic_raw):
    path = tmp_path / "synthetic-raw.fif"
    synthetic_raw.save(path, overwrite=True)
    loaded = load_lfp(
        LoadConfig(path=path, file_format="mne", preload=False, pick_channels=["l1"])
    )
    assert loaded.ch_names == ["l1"]
    assert loaded.preload is False


@pytest.mark.integration
def test_load_lfp_reads_epochs_fif(tmp_path, synthetic_epochs):
    path = tmp_path / "synthetic-epo.fif"
    synthetic_epochs.save(path, overwrite=True)
    loaded = load_lfp(LoadConfig(path=path, file_format="mne", preload=True))
    assert len(loaded) == len(synthetic_epochs)
    assert loaded._data.dtype == np.dtype("float32")


@pytest.mark.integration
def test_baseline_lfp_raw_path(synthetic_raw):
    baselined, summary = baseline_lfp(
        synthetic_raw,
        BaselineConfig(enabled=True, mode="zscore", baseline_window=(0.0, 1.0)),
    )
    assert baselined._data.dtype == np.dtype("float32")
    assert set(summary["channel"]) == set(synthetic_raw.ch_names)


@pytest.mark.integration
def test_preprocess_lfp_none_returns_identity(synthetic_raw):
    out = preprocess_lfp(synthetic_raw, ReferenceConfig(method="none"))
    assert out is synthetic_raw


@pytest.mark.integration
def test_detect_artifacts_none_and_helpers(synthetic_raw):
    tables = detect_artifacts(synthetic_raw, ArtifactConfig(methods=["none"]))
    assert "none" in tables
    assert tables["none"].empty
    arr = _get_data_array(synthetic_raw, copy=True)
    assert arr.dtype == np.dtype("float32")
    assert _ensure_preloaded(synthetic_raw) is synthetic_raw
    assert _downcast_mne_data(synthetic_raw)._data.dtype == np.dtype("float32")


@pytest.mark.integration
def test_compute_spectral_psd(synthetic_raw):
    result = compute_spectral_features(
        synthetic_raw,
        SpectralConfig(enabled=True, method="psd", fmin=1.0, fmax=40.0),
    )
    assert result["method"] == "psd"
    assert "spectrum" in result


@pytest.mark.integration
def test_make_epochs_with_metadata_and_disabled(synthetic_raw):
    assert make_epochs(synthetic_raw, EpochConfig(enabled=False)) is None
    epochs = make_epochs(
        synthetic_raw,
        EpochConfig(
            enabled=True,
            event_name="demo",
            event_times=[5.0, 10.0],
            tmin=-0.2,
            tmax=0.5,
            metadata={"trial": [1, 2]},
        ),
    )
    assert len(epochs) == 2
    assert list(epochs.metadata["trial"]) == [1, 2]


@pytest.mark.integration
def test_make_epochs_metadata_mismatch_raises(synthetic_raw):
    with pytest.raises(ConfigurationError):
        make_epochs(
            synthetic_raw,
            EpochConfig(
                enabled=True,
                event_name="demo",
                event_times=[5.0, 10.0],
                metadata={"trial": [1]},
            ),
        )


@pytest.mark.integration
def test_run_pipeline_with_baseline_and_psd(synthetic_raw):
    config = PipelineConfig(
        load=LoadConfig(path=synthetic_raw, file_format="mne"),
        reference=ReferenceConfig(method="none"),
        artifact=ArtifactConfig(methods=["none"]),
        baseline=BaselineConfig(enabled=True, mode="mean", baseline_window=(0.0, 0.5)),
        epoch=EpochConfig(enabled=False),
        spectral=SpectralConfig(enabled=True, method="psd", fmin=1.0, fmax=30.0),
    )
    result = run_pipeline(config)
    assert result.epochs is None
    assert result.referenced is not None
    assert result.spectral["method"] == "psd"
    assert not result.baseline_summary.empty


@pytest.mark.integration
def test_detect_artifacts_custom_requires_detector(synthetic_raw):
    with pytest.raises(ConfigurationError):
        detect_artifacts(synthetic_raw, ArtifactConfig(methods=["custom"]))


@pytest.mark.integration
def test_baseline_window_required(synthetic_raw):
    with pytest.raises(ConfigurationError):
        baseline_lfp(synthetic_raw, BaselineConfig(enabled=True, mode="zscore", baseline_window=None))


@pytest.mark.integration
def test_builder_helpers_cover_edge_paths():
    basic = build_basic_pipeline_config("x.fif", artifact_methods=[])
    assert basic.artifact.methods == ["none"]
    spectral = build_spectral_pipeline_config(
        "x.fif",
        spectral_method="psd",
        baseline_mode="zscore",
        event_name="ev",
        event_times=[1.0],
    )
    assert spectral.spectral.enabled is True
    assert spectral.epoch.enabled is True
    assert spectral.baseline.enabled is True
