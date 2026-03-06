"""Project-specific exceptions for beginner-facing workflow APIs."""


class LFPAnalysisError(Exception):
    """Base exception for LFPAnalysis workflow failures."""


class ConfigurationError(LFPAnalysisError):
    """Raised when a user-provided configuration is invalid or incomplete."""


class DataContractError(LFPAnalysisError):
    """Raised when input files do not satisfy documented schema requirements."""


class MissingDependencyError(LFPAnalysisError):
    """Raised when an optional dependency is required for the requested workflow."""
