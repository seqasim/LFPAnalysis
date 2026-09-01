"""Integration tests for the stable workflow API using synthetic MNE objects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from LFPAnalysis import (
    AnalysisConfig,
    ArtifactConfig,
    BaselineConfig,
    EpochConfig,
    LoadConfig,
    PipelineConfig,
    ReferenceConfig,
    SpectralConfig,
    TfrConfig,
    baseline_lfp,
    build_basic_pipeline_config,
    build_spectral_pipeline_config,
    compute_spectral_features,
    detect_artifacts,
    load_lfp,
    make_epochs,
    preprocess_lfp,
    run_analysis,
    run_pipeline,
)
from LFPAnalysis.exceptions import ConfigurationError
from LFPAnalysis.schemas import ARTIFACT_EVENT_COLUMNS
from LFPAnalysis.workflow import (
    _downcast_mne_data,
    _ensure_preloaded,
    _get_data_array,
    load_electrode_metadata,
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
def test_detect_artifacts_ied_accepts_float32_raw(synthetic_raw):
    """MNE filtering requires float64; IED detection must upcast then restore float32 storage."""
    synthetic_raw._data = np.asarray(synthetic_raw.get_data(), dtype=np.float32)
    assert synthetic_raw._data.dtype == np.float32
    tables = detect_artifacts(synthetic_raw, ArtifactConfig(methods=["ied"]))
    assert "ied" in tables
    assert list(tables["ied"].columns) == list(ARTIFACT_EVENT_COLUMNS) or len(tables["ied"].columns) >= 1
    assert synthetic_raw._data.dtype == np.float32


@pytest.mark.unit
def test_filter_mne_object_upcasts_then_restores_working_dtype(synthetic_raw):
    from LFPAnalysis.mne_compat import filter_mne_object

    synthetic_raw._data = np.asarray(synthetic_raw.get_data(), dtype=np.float32)
    filtered = filter_mne_object(synthetic_raw, 1.0, 40.0, verbose=False)
    assert filtered._data.dtype.name == "float32"
    assert synthetic_raw._data.dtype.name == "float32"


@pytest.mark.unit
def test_downcast_mne_data_restores_dtype_when_preload_false():
    """Upcast/downcast must not skip restore just because preload is False."""
    from types import SimpleNamespace

    from LFPAnalysis.mne_compat import downcast_mne_data, upcast_mne_data

    fake = SimpleNamespace(preload=False, _data=np.ones((2, 10), dtype=np.float32))
    upcast_mne_data(fake)
    assert fake._data.dtype == np.float64
    downcast_mne_data(fake)
    assert fake._data.dtype.name == "float32"


@pytest.mark.unit
def test_filter_array_upcasts_then_restores_working_dtype():
    from LFPAnalysis.mne_compat import filter_array

    data = np.random.randn(2, 2000).astype(np.float32)
    filtered = filter_array(data, 200.0, l_freq=1.0, h_freq=40.0, verbose=False)
    assert filtered.dtype.name == "float32"
    assert filtered.shape == data.shape


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
    assert result.baseline_summary.empty


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
    assert basic.electrode.path is None
    basic_with_elec = build_basic_pipeline_config("x.fif", electrode_path="x_labels.xlsx")
    assert str(basic_with_elec.electrode.path) == "x_labels.xlsx"
    spectral = build_spectral_pipeline_config(
        "x.fif",
        spectral_method="psd",
        baseline_mode="zscore",
        event_name="ev",
        event_times=[1.0],
    )
    assert spectral.spectral.enabled is True
    assert spectral.epoch.enabled is True
    assert spectral.baseline.enabled is False


@pytest.mark.integration
def test_cross_event_baseline_on_sample_data():
    """End-to-end: baseline feedback_start epochs using baseline_start windows."""
    import pandas as pd
    from pathlib import Path

    from LFPAnalysis import build_event_locked_pipeline_config

    root = Path(__file__).resolve().parents[1]
    beh = pd.read_csv(root / "data" / "sample_beh.csv")
    config = build_event_locked_pipeline_config(
        root / "data" / "sample_ieeg_bp.fif",
        file_format="mne",
        event_name="feedback_start",
        event_times=beh["feedback_start"].tolist(),
        baseline_mode="zscore",
        baseline_event_times=beh["baseline_start"].tolist(),
        baseline_window=(-0.5, 0.0),
        tmin=-0.5,
        tmax=1.5,
        metadata={"reward": beh["reward"].tolist()},
    )
    result = run_pipeline(config)
    assert result.epochs is not None
    assert len(result.epochs) == 80
    assert result.metadata["cross_event_baseline"] is True
    assert result.metadata["has_baseline_epochs"] is True
    assert result.baseline_summary.empty


@pytest.mark.integration
def test_run_analysis_keeps_epochs_raw_when_baseline_enabled(synthetic_epochs):
    original = synthetic_epochs.get_data(copy=True)
    analyzed = run_analysis(
        synthetic_epochs,
        AnalysisConfig(
            baseline=BaselineConfig(enabled=True, mode="zscore", baseline_window=(-0.5, 0.0)),
            spectral=SpectralConfig(enabled=False, method="none"),
            tfr=TfrConfig(enabled=False, method="none"),
        ),
    )
    assert np.allclose(analyzed.epochs.get_data(), original)
    assert analyzed.baseline_summary.empty


@pytest.mark.integration
def test_preprocess_lfp_car_accepts_missing_electrode_path(synthetic_raw):
    out = preprocess_lfp(synthetic_raw, ReferenceConfig(method="car"))
    assert out is not synthetic_raw
    assert out._data.dtype == np.float32
    assert np.allclose(out.get_data().mean(axis=0), 0.0, atol=1e-6)


@pytest.mark.integration
def test_preprocess_lfp_car_excludes_bads_from_reference(mne_module):
    sfreq = 100.0
    times = np.arange(1000) / sfreq
    ch1 = np.sin(2 * np.pi * 3 * times)
    ch2 = np.cos(2 * np.pi * 5 * times)
    outlier = np.ones_like(times) * 1000.0
    raw = mne_module.io.RawArray(
        np.vstack([ch1, ch2, outlier]),
        mne_module.create_info(["c1", "c2", "c3"], sfreq, ch_types=["seeg", "seeg", "seeg"]),
        verbose=False,
    )
    raw.info["bads"] = ["c3"]
    reref = preprocess_lfp(raw, ReferenceConfig(method="car"))
    expected_ref = (ch1 + ch2) / 2.0
    reref_data = reref.get_data()
    assert np.allclose(reref_data[0], ch1 - expected_ref, atol=1e-6)
    assert np.allclose(reref_data[1], ch2 - expected_ref, atol=1e-6)
    assert np.allclose(reref_data[2], outlier - expected_ref, atol=1e-6)


@pytest.mark.integration
def test_preprocess_lfp_car_trimmed_matches_trim_mean(mne_module):
    scipy_stats = pytest.importorskip("scipy.stats")
    sfreq = 50.0
    times = np.arange(500) / sfreq
    data = np.vstack(
        [
            np.sin(2 * np.pi * 2 * times),
            np.cos(2 * np.pi * 4 * times),
            np.sin(2 * np.pi * 7 * times),
            np.cos(2 * np.pi * 10 * times),
            np.ones_like(times) * 30.0,
        ]
    )
    raw = mne_module.io.RawArray(
        data,
        mne_module.create_info(["c1", "c2", "c3", "c4", "c5"], sfreq, ch_types=["seeg"] * 5),
        verbose=False,
    )
    reref = preprocess_lfp(raw, ReferenceConfig(method="car_trimmed"))
    expected_ref = scipy_stats.trim_mean(data, proportiontocut=0.2, axis=0)
    recovered_ref = data[0] - reref.get_data()[0]
    assert np.allclose(recovered_ref, expected_ref, atol=1e-6)


@pytest.mark.integration
def test_preprocess_lfp_car_trimmed_reduces_outlier_influence(mne_module):
    sfreq = 20.0
    data = np.vstack(
        [
            np.linspace(-1, 1, 400),
            np.linspace(1, -1, 400),
            np.sin(np.linspace(0, 6, 400)),
            np.cos(np.linspace(0, 6, 400)),
            np.ones(400) * 1e4,
        ]
    )
    raw = mne_module.io.RawArray(
        data,
        mne_module.create_info(["a", "b", "c", "d", "e"], sfreq, ch_types=["seeg"] * 5),
        verbose=False,
    )
    car = preprocess_lfp(raw, ReferenceConfig(method="car"))
    trimmed = preprocess_lfp(raw, ReferenceConfig(method="car_trimmed"))
    car_ref = data[0] - car.get_data()[0]
    trimmed_ref = data[0] - trimmed.get_data()[0]
    assert np.nanmean(np.abs(car_ref)) > np.nanmean(np.abs(trimmed_ref)) * 5


@pytest.mark.integration
def test_bipolar_pipeline_derives_referenced_electrode_sheet(mne_module, monkeypatch, tmp_path):
    electrode_path = tmp_path / "labels.xlsx"
    np.random.seed(0)
    import pandas as pd

    pd.DataFrame(
        {
            "label": ["a1", "a2"],
            "mni_x": [0.0, 10.0],
            "mni_y": [2.0, 6.0],
            "mni_z": [4.0, 8.0],
        }
    ).to_excel(electrode_path, index=False)

    sfreq = 100.0
    times = np.arange(1000) / sfreq
    raw = mne_module.io.RawArray(
        np.vstack([np.sin(times), np.cos(times)]),
        mne_module.create_info(["a1", "a2"], sfreq, ch_types=["seeg", "seeg"]),
        verbose=False,
    )

    def _fake_ref_mne(mne_data, elec_path, method, site):
        reref = mne_data.copy().pick(["a1"])
        reref.rename_channels({"a1": "a1-a2"})
        return reref

    monkeypatch.setattr(
        "LFPAnalysis.workflow._legacy_preprocess_module",
        lambda: SimpleNamespace(ref_mne=_fake_ref_mne),
    )

    config = build_basic_pipeline_config(
        raw,
        file_format="mne",
        reference_method="bipolar",
        electrode_path=electrode_path,
        preload=True,
    )
    result = run_pipeline(config)
    assert result.referenced is not None
    assert result.electrode_df is not None
    assert list(result.electrode_df["label"]) == list(result.referenced.ch_names)
    assert result.metadata["electrode_df_referenced"] is True

    source = load_electrode_metadata(electrode_path).copy()
    source_idx = source.set_index(source["label"].str.lower())
    first_pair = result.referenced.ch_names[0]
    anode, cathode = first_pair.split("-", 1)
    expected_x = (float(source_idx.loc[anode, "mni_x"]) + float(source_idx.loc[cathode, "mni_x"])) / 2.0
    out_row = result.electrode_df.set_index("label").loc[first_pair]
    assert out_row["mni_x"] == pytest.approx(expected_x)


@pytest.mark.integration
def test_car_trimmed_pipeline_keeps_original_electrode_sheet():
    root = Path(__file__).resolve().parents[1]
    config = build_basic_pipeline_config(
        root / "data" / "sample_ieeg.fif",
        file_format="mne",
        reference_method="car_trimmed",
        electrode_path=root / "data" / "sample_labels.xlsx",
        preload=True,
    )
    result = run_pipeline(config)
    assert result.referenced is not None
    assert result.metadata["electrode_df_referenced"] is False
    original = load_electrode_metadata(root / "data" / "sample_labels.xlsx")
    assert result.electrode_df is not None
    assert result.electrode_df.equals(original)
    assert list(result.referenced.ch_names) == list(load_lfp(LoadConfig(path=root / "data" / "sample_ieeg.fif", file_format="mne")).ch_names)
