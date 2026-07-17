"""Tests for the prep / analysis dual-spine API."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from LFPAnalysis import (
    AnalysisConfig,
    BaselineConfig,
    EpochConfig,
    LoadConfig,
    PipelineConfig,
    PrepConfig,
    SpectralConfig,
    SyncConfig,
    TfrConfig,
    analysis_config_from_pipeline,
    build_analysis_config,
    build_prep_config,
    prep_config_from_pipeline,
    run_analysis,
    run_pipeline,
    run_prep,
)
from LFPAnalysis.config import ReferenceConfig


@pytest.fixture
def synthetic_raw(tmp_path):
    sfreq = 100.0
    info = mne.create_info(ch_names=["l1", "l2"], sfreq=sfreq, ch_types="seeg")
    data = np.random.randn(2, int(sfreq * 5.0)).astype(np.float32)
    raw = mne.io.RawArray(data, info)
    path = tmp_path / "synth-raw.fif"
    raw.save(path, overwrite=True)
    return path


@pytest.mark.unit
def test_prep_config_from_pipeline_strips_analysis_fields(synthetic_raw):
    pipeline = PipelineConfig(
        load=LoadConfig(path=synthetic_raw, file_format="mne"),
        baseline=BaselineConfig(mode="zscore", enabled=True, baseline_window=(-0.2, 0.0)),
        spectral=SpectralConfig(enabled=True, method="psd"),
        epoch=EpochConfig(enabled=True, event_times=[1.0, 2.0], tmin=-0.2, tmax=0.5),
    )
    prep = prep_config_from_pipeline(pipeline)
    analysis = analysis_config_from_pipeline(pipeline)
    assert isinstance(prep, PrepConfig)
    assert prep.epoch.enabled
    assert analysis.baseline.enabled
    assert analysis.spectral.method == "psd"
    assert not hasattr(analysis, "sync")


@pytest.mark.unit
def test_run_prep_then_run_analysis(synthetic_raw):
    prep = run_prep(
        PrepConfig(
            load=LoadConfig(path=synthetic_raw, file_format="mne", preload=True),
            epoch=EpochConfig(
                enabled=True,
                event_name="ev",
                event_times=[1.0, 2.0, 3.0],
                tmin=-0.2,
                tmax=0.5,
            ),
        )
    )
    assert prep.epochs is not None
    assert prep.sync == {}
    assert prep.metadata["spine"] == "prep"

    analysis = run_analysis(
        prep.epochs,
        AnalysisConfig(
            baseline=BaselineConfig(mode="zscore", enabled=True, baseline_window=(-0.2, 0.0)),
            spectral=SpectralConfig(enabled=True, method="psd", fmin=1.0, fmax=40.0),
        ),
    )
    assert analysis.metadata["spine"] == "analysis"
    assert "spectrum" in analysis.spectral
    assert analysis.baseline_summary.shape[0] == 2


@pytest.mark.unit
def test_run_pipeline_composes_prep_and_analysis(synthetic_raw):
    result = run_pipeline(
        PipelineConfig(
            load=LoadConfig(path=synthetic_raw, file_format="mne", preload=True),
            epoch=EpochConfig(
                enabled=True,
                event_times=[1.0, 2.0],
                tmin=-0.2,
                tmax=0.5,
            ),
            baseline=BaselineConfig(mode="zscore", enabled=True, baseline_window=(-0.2, 0.0)),
            spectral=SpectralConfig(enabled=True, method="psd", fmin=1.0, fmax=40.0),
        )
    )
    assert result.epochs is not None
    assert result.raw is None
    assert result.referenced is None
    assert "spectrum" in result.spectral


@pytest.mark.unit
def test_run_analysis_tfr_morlet(synthetic_raw):
    prep = run_prep(
        build_prep_config(
            synthetic_raw,
            event_name="ev",
            event_times=[1.0, 2.0],
            tmin=-0.2,
            tmax=0.5,
        )
    )
    # ensure preload for TFR
    prep.epochs.load_data()
    analysis = run_analysis(
        prep.epochs,
        build_analysis_config(
            baseline_mode="none",
            tfr_method="morlet",
            tfr_freqs=[8.0, 12.0, 20.0],
            tfr_n_cycles=3.0,
        ),
    )
    assert analysis.tfr["method"] == "morlet"
    assert "power" in analysis.tfr


@pytest.mark.unit
def test_sync_precomputed_updates_epoch_slope(synthetic_raw):
    prep = run_prep(
        PrepConfig(
            load=LoadConfig(path=synthetic_raw, file_format="mne", preload=True),
            sync=SyncConfig(enabled=True, source="precomputed", slope=2.0, offset=0.5),
            epoch=EpochConfig(
                enabled=True,
                event_times=[0.5, 1.0],  # neural seconds = 2*beh + 0.5 → 1.5, 2.5
                tmin=-0.1,
                tmax=0.2,
            ),
        )
    )
    assert prep.sync["slope"] == 2.0
    assert prep.sync["offset"] == 0.5
    assert prep.epochs is not None
    assert len(prep.epochs) == 2


@pytest.mark.unit
def test_laplacian_no_longer_in_reference_literal():
    from LFPAnalysis.config import ReferenceMethod
    from typing import get_args

    assert "laplacian" not in get_args(ReferenceMethod)


@pytest.mark.unit
def test_advanced_package_lazy_exports():
    from LFPAnalysis import advanced

    assert hasattr(advanced, "make_surrogate_arrays")
    fn = advanced.make_surrogate_arrays
    assert callable(fn)
