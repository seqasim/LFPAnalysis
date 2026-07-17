"""Curated advanced escape hatch for nimble, post-handoff science.

Import algorithms from here. Prep and analysis spines remain the typed beginner
path; this package is for composing connectivity, stats, ROI, and site I/O
freely after you have MNE Epochs.

Mega-module splits into ``advanced.prep`` / ``advanced.connectivity`` etc. land
in later milestones; for now public names lazy-load from existing modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "select_rois_picks",
    "select_picks_rois",
    "make_surrogate_arrays",
    "compute_connectivity",
    "permutation_regression_zscore",
    "time_resolved_mlm",
    "synchronize_data",
    "synchronize_data_robust",
]

_LAZY: dict[str, tuple[str, str]] = {
    "select_rois_picks": ("LFPAnalysis.analysis_utils", "select_rois_picks"),
    "select_picks_rois": ("LFPAnalysis.analysis_utils", "select_picks_rois"),
    "make_surrogate_arrays": ("LFPAnalysis.oscillation_utils", "make_surrogate_arrays"),
    "compute_connectivity": ("LFPAnalysis.oscillation_utils", "compute_connectivity"),
    "permutation_regression_zscore": (
        "LFPAnalysis.statistics_utils",
        "permutation_regression_zscore",
    ),
    "time_resolved_mlm": ("LFPAnalysis.statistics_utils", "time_resolved_mlm"),
    "synchronize_data": ("LFPAnalysis.sync_utils", "synchronize_data"),
    "synchronize_data_robust": ("LFPAnalysis.sync_utils", "synchronize_data_robust"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value
