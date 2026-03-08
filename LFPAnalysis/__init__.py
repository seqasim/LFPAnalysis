"""Stable public exports for the beginner-facing LFPAnalysis workflow API."""

from .builders import (
    build_basic_pipeline_config,
    build_event_locked_pipeline_config,
    build_spectral_pipeline_config,
)
from .config import (
    ArtifactConfig,
    BaselineConfig,
    EpochConfig,
    LoadConfig,
    PipelineConfig,
    ReferenceConfig,
    SpectralConfig,
)
from .exceptions import (
    ConfigurationError,
    DataContractError,
    LFPAnalysisError,
    MissingDependencyError,
)
from .results import PipelineResult
from .schemas import (
    ARTIFACT_EVENT_COLUMNS,
    BASELINE_SUMMARY_COLUMNS,
    ELECTRODE_OPTIONAL_COLUMNS,
    ELECTRODE_REQUIRED_COLUMNS,
)
from .workflow import (
    baseline_lfp,
    compute_spectral_features,
    detect_artifacts,
    load_electrode_metadata,
    load_lfp,
    make_epochs,
    preprocess_lfp,
    run_pipeline,
)

__all__ = [
    "ARTIFACT_EVENT_COLUMNS",
    "ArtifactConfig",
    "BASELINE_SUMMARY_COLUMNS",
    "BaselineConfig",
    "build_basic_pipeline_config",
    "build_event_locked_pipeline_config",
    "build_spectral_pipeline_config",
    "ConfigurationError",
    "DataContractError",
    "ELECTRODE_OPTIONAL_COLUMNS",
    "ELECTRODE_REQUIRED_COLUMNS",
    "EpochConfig",
    "LFPAnalysisError",
    "LoadConfig",
    "MissingDependencyError",
    "PipelineConfig",
    "PipelineResult",
    "ReferenceConfig",
    "SpectralConfig",
    "baseline_lfp",
    "compute_spectral_features",
    "detect_artifacts",
    "load_electrode_metadata",
    "load_lfp",
    "make_epochs",
    "preprocess_lfp",
    "run_pipeline",
]
