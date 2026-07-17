"""Typed configuration objects for the stable prep and analysis spines."""

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
# laplacian intentionally omitted until implemented (was a reserved registry trap).
ReferenceMethod = Literal["none", "bipolar", "wm"]
SpectralMethod = Literal["none", "psd", "fooof"]
TfrMethod = Literal["none", "morlet"]
SyncSource = Literal["none", "photodiode", "ttl", "precomputed"]
InputFormat = Literal["edf", "neuralynx", "mne"]

# Default working dtype name for signal / TFR arrays. float32 halves RAM vs float64
# while remaining sufficient for typical iEEG/LFP analysis on local machines.
# Resolved to a NumPy dtype at use sites via ``numpy.dtype(WORKING_DTYPE)``.
WORKING_DTYPE = "float32"


@dataclass(slots=True)
class LoadConfig:
    """Configuration for loading continuous or epoched data."""

    path: PathLike | Sequence[PathLike] | Any
    file_format: InputFormat = "mne"
    preload: bool = False
    memmap: bool = True
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
    """Configuration for signal re-referencing (prep spine)."""

    method: ReferenceMethod = "none"
    electrode_path: PathLike | None = None
    site: str = "MSSM"


@dataclass(slots=True)
class ArtifactConfig:
    """Configuration for continuous artifact detection (prep spine)."""

    methods: list[ArtifactMethod] = field(default_factory=lambda: ["none"])
    misc_peak_thresh: float = 6.0
    ied_peak_thresh: float = 5.0
    ied_closeness_thresh: float = 0.25
    ied_width_thresh: float = 0.2
    custom_detector: Callable[[Any], dict[str, Any]] | None = None
    write_sidecars: bool = False


@dataclass(slots=True)
class SyncConfig:
    """Behavioral–neural synchronization (prep spine only).

    Analysis never re-runs sync; slope/offset are provenance on the Epochs handoff.
    """

    enabled: bool = False
    source: SyncSource = "none"
    behav_times: list[float] = field(default_factory=list)
    sync_channel: str | None = None
    nev_data: dict[str, Any] | None = None
    smooth_size: int = 11
    wind_size: int = 15
    height: float = 0.5
    use_robust: bool = False
    # When source='precomputed', these are required.
    slope: float | None = None
    offset: float | None = None


@dataclass(slots=True)
class ElectrodeConfig:
    """Electrode / anatomy table consumption (prep spine; not localization)."""

    path: PathLike | None = None
    site: str = "MSSM"
    load_into_result: bool = True


@dataclass(slots=True)
class BaselineConfig:
    """Configuration for baselining continuous or epoched data (analysis spine)."""

    mode: BaselineMode = "none"
    enabled: bool = False
    baseline_window: tuple[float, float] | None = None


@dataclass(slots=True)
class EpochConfig:
    """Configuration for epoch extraction around behavioral timestamps (prep spine)."""

    enabled: bool = False
    event_name: str = "task_event"
    event_times: list[float] = field(default_factory=list)
    slope: float = 1.0
    offset: float = 0.0
    tmin: float = -0.5
    tmax: float = 1.5
    buffer_s: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class SpectralConfig:
    """Configuration for optional spectral feature computation (analysis spine)."""

    enabled: bool = False
    method: SpectralMethod = "none"
    fmin: float = 1.0
    fmax: float = 150.0
    n_fft: int | None = None
    fooof_range: tuple[float, float] = (1.0, 40.0)
    fooof_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TfrConfig:
    """Configuration for optional time-frequency features (analysis spine)."""

    enabled: bool = False
    method: TfrMethod = "none"
    freqs: list[float] | Any | None = None
    n_cycles: float | Any = 7.0
    baseline_mode: BaselineMode = "zscore"
    apply_baseline: bool = True
    decim: int = 1
    n_jobs: int = 1


@dataclass(slots=True)
class PrepConfig:
    """Prep spine: raw clinical files → event-locked MNE Epochs (+ metadata).

    Sync, electrode tables, referencing, and continuous artifact detection live here
    so they can evolve without rewriting analysis.
    """

    load: LoadConfig
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    electrode: ElectrodeConfig = field(default_factory=ElectrodeConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)


@dataclass(slots=True)
class AnalysisConfig:
    """Analysis spine: starts from MNE Epochs (no sync / localization)."""

    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    tfr: TfrConfig = field(default_factory=TfrConfig)


@dataclass(slots=True)
class PipelineConfig:
    """Tutorial convenience: prep fields + analysis fields in one object.

    Prefer :class:`PrepConfig` / :class:`AnalysisConfig` for new code. ``run_pipeline``
    still accepts this flat config and runs prep then analysis.
    """

    load: LoadConfig
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    electrode: ElectrodeConfig = field(default_factory=ElectrodeConfig)
    tfr: TfrConfig = field(default_factory=TfrConfig)


def prep_config_from_pipeline(config: PipelineConfig) -> PrepConfig:
    """Extract the prep spine from a flat :class:`PipelineConfig`."""
    return PrepConfig(
        load=config.load,
        reference=config.reference,
        artifact=config.artifact,
        sync=config.sync,
        electrode=config.electrode,
        epoch=config.epoch,
    )


def analysis_config_from_pipeline(config: PipelineConfig) -> AnalysisConfig:
    """Extract the analysis spine from a flat :class:`PipelineConfig`."""
    return AnalysisConfig(
        baseline=config.baseline,
        spectral=config.spectral,
        tfr=config.tfr,
    )
