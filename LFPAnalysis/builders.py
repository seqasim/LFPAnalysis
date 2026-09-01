"""Convenience constructors for prep, analysis, and tutorial pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

from .config import (
    AnalysisConfig,
    ArtifactConfig,
    BaselineConfig,
    ElectrodeConfig,
    EpochConfig,
    LoadConfig,
    PipelineConfig,
    PrepConfig,
    ReferenceConfig,
    SpectralConfig,
    SyncConfig,
    TfrConfig,
)
from .exceptions import ConfigurationError


PathLike = str | Path


def _normalize_methods(methods: Iterable[str] | None, default: list[str]) -> list[str]:
    if methods is None:
        return default
    normalized = [str(method).strip() for method in methods if str(method).strip()]
    return normalized or default


def build_prep_config(
    path: PathLike,
    *,
    file_format: str = "mne",
    reference_method: str = "none",
    electrode_path: PathLike | None = None,
    electrode_site: str = "MSSM",
    artifact_methods: Iterable[str] | None = None,
    resample_sfreq: float | None | str = "auto",
    notch_freqs: tuple[float, ...] | None = (60.0, 120.0, 180.0, 240.0),
    check_bad_channels: bool = False,
    include_micros: bool = False,
    preload: bool = False,
    event_name: str | None = None,
    event_times: list[float] | None = None,
    tmin: float = -0.5,
    tmax: float = 1.5,
    slope: float = 1.0,
    offset: float = 0.0,
    buffer_s: float = 1.0,
    sync: SyncConfig | None = None,
    metadata: dict | None = None,
    baseline_event_times: list[float] | None = None,
    baseline_window: tuple[float, float] | None = None,
) -> PrepConfig:
    """Build a prep-spine configuration (raw → Epochs handoff).

    When ``baseline_event_times`` is provided, prep also extracts baseline
    epochs locked to those times. ``baseline_window`` then becomes the
    ``(baseline_tmin, baseline_tmax)`` window relative to each baseline event.
    """
    epoch_enabled = bool(event_name and event_times)
    baseline_tmin = None
    baseline_tmax = None
    if baseline_event_times is not None:
        if not event_times:
            raise ConfigurationError(
                "baseline_event_times requires event_times (task events) to be set."
            )
        if len(baseline_event_times) != len(event_times):
            raise ConfigurationError(
                "baseline_event_times must have the same length as event_times."
            )
        if baseline_window is None:
            raise ConfigurationError(
                "baseline_window is required when baseline_event_times is set."
            )
        baseline_tmin, baseline_tmax = float(baseline_window[0]), float(baseline_window[1])
    return PrepConfig(
        load=LoadConfig(
            path=path,
            file_format=file_format,
            preload=preload,
            resample_sfreq=resample_sfreq,
            notch_freqs=notch_freqs,
            check_bad_channels=check_bad_channels,
            include_micros=include_micros,
        ),
        reference=ReferenceConfig(
            method=reference_method,
            electrode_path=electrode_path,
            site=electrode_site,
        ),
        artifact=ArtifactConfig(methods=_normalize_methods(artifact_methods, ["none"])),
        sync=sync if sync is not None else SyncConfig(),
        electrode=ElectrodeConfig(path=electrode_path, site=electrode_site),
        epoch=EpochConfig(
            enabled=epoch_enabled,
            event_name=event_name or "task_event",
            event_times=event_times or [],
            slope=slope,
            offset=offset,
            tmin=tmin,
            tmax=tmax,
            buffer_s=buffer_s,
            metadata=metadata,
            baseline_event_times=baseline_event_times,
            baseline_tmin=baseline_tmin,
            baseline_tmax=baseline_tmax,
        ),
    )


def build_analysis_config(
    *,
    baseline_mode: str = "none",
    baseline_window: tuple[float, float] | None = None,
    spectral_method: str = "none",
    fmin: float = 1.0,
    fmax: float = 150.0,
    fooof_range: tuple[float, float] = (1.0, 40.0),
    tfr_method: str = "none",
    tfr_freqs: list[float] | None = None,
    tfr_n_cycles: float = 7.0,
) -> AnalysisConfig:
    """Build an analysis-spine configuration (Epochs → features)."""
    if baseline_mode != "none" or baseline_window is not None:
        warnings.warn(
            "build_analysis_config baseline args now control TFR baseline only; "
            "voltage baselining in run_analysis is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
    if spectral_method not in {"none", "psd", "fooof"}:
        raise ConfigurationError(
            "build_analysis_config supports spectral methods: none, psd, fooof."
        )
    if tfr_method not in {"none", "morlet"}:
        raise ConfigurationError("build_analysis_config supports tfr methods: none, morlet.")
    spectral_enabled = spectral_method != "none"
    tfr_enabled = tfr_method != "none"
    return AnalysisConfig(
        baseline=BaselineConfig(mode="none", enabled=False, baseline_window=None),
        spectral=SpectralConfig(
            enabled=spectral_enabled,
            method=spectral_method if spectral_enabled else "none",
            fmin=fmin,
            fmax=fmax,
            fooof_range=fooof_range,
        ),
        tfr=TfrConfig(
            enabled=tfr_enabled,
            method=tfr_method if tfr_enabled else "none",
            freqs=tfr_freqs,
            n_cycles=tfr_n_cycles,
            baseline_mode=baseline_mode,
            apply_baseline=baseline_mode != "none",
        ),
    )


def build_tutorial_pipeline_config(
    path: PathLike,
    *,
    file_format: str = "mne",
    reference_method: str = "none",
    electrode_path: PathLike | None = None,
    artifact_methods: Iterable[str] | None = None,
    preload: bool = False,
    event_name: str | None = None,
    event_times: list[float] | None = None,
    tmin: float = -0.5,
    tmax: float = 1.5,
    slope: float = 1.0,
    offset: float = 0.0,
    baseline_mode: str = "zscore",
    baseline_window: tuple[float, float] = (-0.5, 0.0),
    spectral_method: str = "none",
    tfr_method: str = "none",
    tfr_freqs: list[float] | None = None,
    buffer_s: float = 1.0,
    resample_sfreq: float | None | str = "auto",
    notch_freqs: tuple[float, ...] | None = (60.0, 120.0, 180.0, 240.0),
    check_bad_channels: bool = False,
) -> PipelineConfig:
    """Compose prep + analysis into a flat PipelineConfig for tutorials."""
    prep = build_prep_config(
        path,
        file_format=file_format,
        reference_method=reference_method,
        electrode_path=electrode_path,
        artifact_methods=artifact_methods,
        preload=preload,
        resample_sfreq=resample_sfreq,
        notch_freqs=notch_freqs,
        check_bad_channels=check_bad_channels,
        event_name=event_name,
        event_times=event_times,
        tmin=tmin,
        tmax=tmax,
        slope=slope,
        offset=offset,
        buffer_s=buffer_s,
    )
    analysis = build_analysis_config(
        baseline_mode=baseline_mode if event_name and event_times else "none",
        baseline_window=baseline_window if event_name and event_times else None,
        spectral_method=spectral_method,
        tfr_method=tfr_method,
        tfr_freqs=tfr_freqs,
    )
    return PipelineConfig(
        load=prep.load,
        reference=prep.reference,
        artifact=prep.artifact,
        sync=prep.sync,
        electrode=prep.electrode,
        epoch=prep.epoch,
        baseline=analysis.baseline,
        spectral=analysis.spectral,
        tfr=analysis.tfr,
    )


def build_basic_pipeline_config(
    path: PathLike,
    *,
    file_format: str = "mne",
    reference_method: str = "none",
    electrode_path: PathLike | None = None,
    artifact_methods: Iterable[str] | None = None,
    resample_sfreq: float | None | str = "auto",
    notch_freqs: tuple[float, ...] | None = (60.0, 120.0, 180.0, 240.0),
    check_bad_channels: bool = False,
    include_micros: bool = False,
    preload: bool = False,
) -> PipelineConfig:
    """Build the simplest valid pipeline configuration for first-time users."""
    return PipelineConfig(
        load=LoadConfig(
            path=path,
            file_format=file_format,
            preload=preload,
            resample_sfreq=resample_sfreq,
            notch_freqs=notch_freqs,
            check_bad_channels=check_bad_channels,
            include_micros=include_micros,
        ),
        reference=ReferenceConfig(method=reference_method, electrode_path=electrode_path),
        artifact=ArtifactConfig(methods=_normalize_methods(artifact_methods, ["none"])),
        baseline=BaselineConfig(mode="none", enabled=False),
        epoch=EpochConfig(enabled=False),
        spectral=SpectralConfig(enabled=False, method="none"),
        electrode=ElectrodeConfig(path=electrode_path),
    )


def build_event_locked_pipeline_config(
    path: PathLike,
    *,
    event_name: str,
    event_times: list[float],
    file_format: str = "mne",
    reference_method: str = "none",
    electrode_path: PathLike | None = None,
    artifact_methods: Iterable[str] | None = None,
    preload: bool = False,
    baseline_mode: str = "zscore",
    baseline_window: tuple[float, float] = (-0.5, 0.0),
    baseline_event_times: list[float] | None = None,
    tmin: float = -0.5,
    tmax: float = 1.5,
    slope: float = 1.0,
    offset: float = 0.0,
    metadata: dict | None = None,
    buffer_s: float = 1.0,
    tfr_method: str = "none",
    tfr_freqs: list[float] | None = None,
    tfr_n_cycles: float = 7.0,
) -> PipelineConfig:
    """Build a beginner-friendly event-locked pipeline configuration.

    Same-event baselining (default)
        ``baseline_window`` is relative to ``event_name`` / ``event_times`` and
        is applied on the task epochs' time axis.

    Cross-event baselining
        Pass ``baseline_event_times`` (same length as ``event_times``). Then
        ``baseline_window`` is interpreted relative to each baseline event
        (e.g. baseline ``recog_time`` epochs using a window around
        ``baseline_time_mem``).
    """
    baseline_tmin = None
    baseline_tmax = None
    if baseline_event_times is not None:
        if len(baseline_event_times) != len(event_times):
            raise ConfigurationError(
                "baseline_event_times must have the same length as event_times."
            )
        baseline_tmin, baseline_tmax = float(baseline_window[0]), float(baseline_window[1])
    return PipelineConfig(
        load=LoadConfig(path=path, file_format=file_format, preload=preload),
        reference=ReferenceConfig(method=reference_method, electrode_path=electrode_path),
        artifact=ArtifactConfig(methods=_normalize_methods(artifact_methods, ["misc"])),
        baseline=BaselineConfig(mode="none", enabled=False, baseline_window=None),
        epoch=EpochConfig(
            enabled=True,
            event_name=event_name,
            event_times=event_times,
            slope=slope,
            offset=offset,
            tmin=tmin,
            tmax=tmax,
            buffer_s=buffer_s,
            metadata=metadata,
            baseline_event_times=baseline_event_times,
            baseline_tmin=baseline_tmin,
            baseline_tmax=baseline_tmax,
        ),
        spectral=SpectralConfig(enabled=False, method="none"),
        tfr=TfrConfig(
            enabled=tfr_method != "none",
            method=tfr_method if tfr_method != "none" else "none",
            freqs=tfr_freqs,
            n_cycles=tfr_n_cycles,
            baseline_mode=baseline_mode,
            apply_baseline=baseline_mode != "none",
            crop_tmin=tmin,
            crop_tmax=tmax,
        ),
        electrode=ElectrodeConfig(path=electrode_path),
    )


def build_spectral_pipeline_config(
    path: PathLike,
    *,
    spectral_method: str = "psd",
    file_format: str = "mne",
    reference_method: str = "none",
    electrode_path: PathLike | None = None,
    artifact_methods: Iterable[str] | None = None,
    preload: bool = False,
    baseline_mode: str = "none",
    baseline_window: tuple[float, float] | None = None,
    event_name: str | None = None,
    event_times: list[float] | None = None,
    tmin: float = -0.5,
    tmax: float = 1.5,
    fmin: float = 1.0,
    fmax: float = 150.0,
    fooof_range: tuple[float, float] = (1.0, 40.0),
) -> PipelineConfig:
    """Build a spectral-analysis starter configuration.

    This helper intentionally supports the stable workflow surface only. For
    connectivity, continue with ``LFPAnalysis.advanced`` after prep/analysis.

    For baseline-only teaching on existing Epochs, prefer ``build_analysis_config``
    + ``run_analysis`` instead of this helper (which always enables spectral).
    """
    epoch_enabled = bool(event_name and event_times)
    if spectral_method not in {"psd", "fooof"}:
        raise ConfigurationError(
            "build_spectral_pipeline_config supports only the stable spectral methods: psd and fooof. "
            "Use the worked examples for TFR and connectivity preparation."
        )
    effective_baseline_window = baseline_window
    if baseline_mode != "none" and effective_baseline_window is None:
        effective_baseline_window = (-0.5, 0.0) if epoch_enabled else None
    return PipelineConfig(
        load=LoadConfig(path=path, file_format=file_format, preload=preload),
        reference=ReferenceConfig(method=reference_method, electrode_path=electrode_path),
        artifact=ArtifactConfig(methods=_normalize_methods(artifact_methods, ["misc"])),
        baseline=BaselineConfig(mode="none", enabled=False, baseline_window=None),
        epoch=EpochConfig(
            enabled=epoch_enabled,
            event_name=event_name or "task_event",
            event_times=event_times or [],
            tmin=tmin,
            tmax=tmax,
        ),
        spectral=SpectralConfig(
            enabled=True,
            method=spectral_method,
            fmin=fmin,
            fmax=fmax,
            fooof_range=fooof_range,
        ),
        electrode=ElectrodeConfig(path=electrode_path),
    )
