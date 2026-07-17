"""Typed result containers returned by the stable prep and analysis spines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .schemas import empty_baseline_summary, empty_sync_summary


@dataclass(slots=True)
class PrepResult:
    """Handoff from prep: Epochs plus replaceable prep metadata.

    Analysis should depend on ``epochs`` (and optionally ``electrode_df`` /
    ``artifact_tables`` / ``sync`` provenance) — not on how sync or electrodes
    were obtained.
    """

    epochs: Any | None
    raw: Any | None = None
    referenced: Any | None = None
    artifact_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    electrode_df: pd.DataFrame | None = None
    sync: dict[str, Any] = field(default_factory=dict)
    sync_summary: pd.DataFrame = field(default_factory=empty_sync_summary)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisResult:
    """Outputs of the analysis spine starting from MNE Epochs."""

    epochs: Any | None = None
    baseline_summary: pd.DataFrame = field(default_factory=empty_baseline_summary)
    spectral: dict[str, Any] = field(default_factory=dict)
    tfr: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    """Tutorial convenience container composing prep + analysis outputs."""

    raw: Any
    referenced: Any
    epochs: Any | None = None
    artifact_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    baseline_summary: pd.DataFrame = field(default_factory=empty_baseline_summary)
    spectral: dict[str, Any] = field(default_factory=dict)
    tfr: dict[str, Any] = field(default_factory=dict)
    electrode_df: pd.DataFrame | None = None
    sync: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
