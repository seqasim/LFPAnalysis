"""Typed configuration objects for the stable workflow layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

PathLike = str | Path
ArtifactMethod = Literal["none", "misc", "ied", "custom"]
BaselineMode = Literal[
    "none",
    "mean",
    "ratio",
    "percent",
    "zscore",
    "logratio",
    "zlogratio",
    "trialwise",
    "continuous",
]
ReferenceMethod = Literal["none", "bipolar", "wm", "laplacian"]
SpectralMethod = Literal["none", "psd", "fooof"]
InputFormat = Literal["edf", "neuralynx", "mne"]


@dataclass(slots=True)
class LoadConfig:
    """Configuration for loading continuous or epoched data."""

    path: PathLike | Sequence[PathLike] | Any
    file_format: InputFormat = "mne"
    preload: bool = True
    resample_sfreq: float | None = None
    include_micros: bool = False
    eeg_names: list[str] = field(default_factory=list)
    resp_names: list[str] = field(default_factory=list)
    ekg_names: list[str] = field(default_factory=list)
    seeg_names: list[str] = field(default_factory=list)
    drop_names: list[str] = field(default_factory=list)
    pick_channels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReferenceConfig:
    """Configuration for signal re-referencing."""

    method: ReferenceMethod = "none"
    electrode_path: PathLike | None = None
    site: str = "MSSM"


@dataclass(slots=True)
class ArtifactConfig:
    """Configuration for artifact detection."""

    methods: list[ArtifactMethod] = field(default_factory=lambda: ["none"])
    misc_peak_thresh: float = 6.0
    ied_peak_thresh: float = 5.0
    ied_closeness_thresh: float = 0.25
    ied_width_thresh: float = 0.2
    custom_detector: Callable[[Any], dict[str, Any]] | None = None


@dataclass(slots=True)
class BaselineConfig:
    """Configuration for baselining continuous or epoched data."""

    mode: BaselineMode = "none"
    enabled: bool = False
    baseline_window: tuple[float, float] | None = None


@dataclass(slots=True)
class EpochConfig:
    """Configuration for epoch extraction around behavioral timestamps."""

    enabled: bool = False
    event_name: str = "task_event"
    event_times: list[float] = field(default_factory=list)
    slope: float = 1.0
    offset: float = 0.0
    tmin: float = -0.5
    tmax: float = 1.5
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class SpectralConfig:
    """Configuration for optional spectral feature computation."""

    enabled: bool = False
    method: SpectralMethod = "none"
    fmin: float = 1.0
    fmax: float = 150.0
    n_fft: int | None = None
    fooof_range: tuple[float, float] = (1.0, 40.0)
    fooof_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineConfig:
    """Top-level workflow configuration."""

    load: LoadConfig
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
