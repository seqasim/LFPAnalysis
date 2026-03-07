"""Integration tests for the stable workflow API using synthetic MNE objects."""

from __future__ import annotations

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
    load_lfp,
    make_epochs,
    run_pipeline,
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
    assert result.raw is not None
    assert result.referenced is not None
    assert result.epochs is not None
    assert "custom" in result.artifact_tables
    assert result.metadata["reference_method"] == "none"
