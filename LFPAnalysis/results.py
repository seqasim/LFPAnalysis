"""Typed result containers returned by the stable workflow API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .schemas import empty_baseline_summary


@dataclass(slots=True)
class PipelineResult:
    """Container for staged workflow outputs."""

    raw: Any
    referenced: Any
    epochs: Any | None = None
    artifact_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    baseline_summary: pd.DataFrame = field(default_factory=empty_baseline_summary)
    spectral: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
