"""Convenience constructors for common beginner-facing pipeline setups."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import (
    ArtifactConfig,
    BaselineConfig,
    EpochConfig,
    LoadConfig,
    PipelineConfig,
    ReferenceConfig,
    SpectralConfig,
)
from .exceptions import ConfigurationError


PathLike = str | Path


def _normalize_methods(methods: Iterable[str] | None, default: list[str]) -> list[str]:
    if methods is None:
        return default
    normalized = [str(method).strip() for method in methods if str(method).strip()]
    return normalized or default


def build_basic_pipeline_config(
    path: PathLike,
    *,
    file_format: str = "mne",
    reference_method: str = "none",
    electrode_path: PathLike | None = None,
    artifact_methods: Iterable[str] | None = None,
    resample_sfreq: float | None = None,
    include_micros: bool = False,
) -> PipelineConfig:
    """Build the simplest valid pipeline configuration for first-time users."""
    return PipelineConfig(
        load=LoadConfig(
            path=path,
            file_format=file_format,
            resample_sfreq=resample_sfreq,
            include_micros=include_micros,
        ),
        reference=ReferenceConfig(method=reference_method, electrode_path=electrode_path),
        artifact=ArtifactConfig(methods=_normalize_methods(artifact_methods, ["none"])),
        baseline=BaselineConfig(mode="none", enabled=False),
        epoch=EpochConfig(enabled=False),
        spectral=SpectralConfig(enabled=False, method="none"),
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
    baseline_mode: str = "zscore",
    baseline_window: tuple[float, float] = (-0.5, 0.0),
    tmin: float = -0.5,
    tmax: float = 1.0,
    slope: float = 1.0,
    offset: float = 0.0,
    metadata: dict | None = None,
) -> PipelineConfig:
    """Build a beginner-friendly event-locked pipeline configuration."""
    return PipelineConfig(
        load=LoadConfig(path=path, file_format=file_format),
        reference=ReferenceConfig(method=reference_method, electrode_path=electrode_path),
        artifact=ArtifactConfig(methods=_normalize_methods(artifact_methods, ["misc"])),
        baseline=BaselineConfig(mode=baseline_mode, enabled=True, baseline_window=baseline_window),
        epoch=EpochConfig(
            enabled=True,
            event_name=event_name,
            event_times=event_times,
            slope=slope,
            offset=offset,
            tmin=tmin,
            tmax=tmax,
            metadata=metadata,
        ),
        spectral=SpectralConfig(enabled=False, method="none"),
    )



def build_spectral_pipeline_config(
    path: PathLike,
    *,
    spectral_method: str = "psd",
    file_format: str = "mne",
    reference_method: str = "none",
    electrode_path: PathLike | None = None,
    artifact_methods: Iterable[str] | None = None,
    baseline_mode: str = "none",
    baseline_window: tuple[float, float] | None = None,
    event_name: str | None = None,
    event_times: list[float] | None = None,
    tmin: float = -0.5,
    tmax: float = 1.0,
    fmin: float = 1.0,
    fmax: float = 150.0,
    fooof_range: tuple[float, float] = (1.0, 40.0),
) -> PipelineConfig:
    """Build a spectral-analysis starter configuration.

    This helper intentionally supports the stable workflow surface only. For TFR and
    connectivity, users should use this preprocessing starter and then continue with
    the advanced utility modules documented in the worked examples.
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
        load=LoadConfig(path=path, file_format=file_format),
        reference=ReferenceConfig(method=reference_method, electrode_path=electrode_path),
        artifact=ArtifactConfig(methods=_normalize_methods(artifact_methods, ["misc"])),
        baseline=BaselineConfig(
            mode=baseline_mode,
            enabled=baseline_mode != "none",
            baseline_window=effective_baseline_window,
        ),
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
    )
