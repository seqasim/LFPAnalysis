"""Shared table schemas and dataframe builders for workflow outputs."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ELECTRODE_REQUIRED_COLUMNS: tuple[str, ...] = ("label",)
ELECTRODE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "NMM",
    "BN246",
    "YBA_1",
    "collapsed_manual",
)
ARTIFACT_EVENT_COLUMNS: tuple[str, ...] = (
    "event_kind",
    "channel",
    "time_seconds",
    "sample_index",
)
BASELINE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "target",
    "channel",
    "mode",
    "baseline_start",
    "baseline_stop",
    "baseline_mean",
    "baseline_std",
)


def empty_event_table() -> pd.DataFrame:
    """Create an empty artifact-event table with the standard schema."""
    return pd.DataFrame(columns=ARTIFACT_EVENT_COLUMNS)


def empty_baseline_summary() -> pd.DataFrame:
    """Create an empty baseline summary table with the standard schema."""
    return pd.DataFrame(columns=BASELINE_SUMMARY_COLUMNS)


def build_event_table(
    channel_events: Mapping[str, Sequence[float] | np.ndarray],
    *,
    event_kind: str,
    sfreq: float,
) -> pd.DataFrame:
    """Convert channel-event mappings into a standard long-form table."""
    records: list[dict[str, object]] = []
    for channel, times in channel_events.items():
        if times is None or (isinstance(times, float) and np.isnan(times)):
            continue
        for time_value in np.asarray(times, dtype=float).tolist():
            records.append(
                {
                    "event_kind": event_kind,
                    "channel": channel,
                    "time_seconds": float(time_value),
                    "sample_index": int(round(float(time_value) * sfreq)),
                }
            )
    if not records:
        return empty_event_table()
    return pd.DataFrame.from_records(records, columns=ARTIFACT_EVENT_COLUMNS)


def build_baseline_summary(
    *,
    target: str,
    channel_names: Iterable[str],
    mode: str,
    baseline_start: float | None,
    baseline_stop: float | None,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
) -> pd.DataFrame:
    """Create a standard per-channel baseline summary table."""
    rows = []
    for channel, mean_value, std_value in zip(channel_names, baseline_mean, baseline_std):
        rows.append(
            {
                "target": target,
                "channel": channel,
                "mode": mode,
                "baseline_start": baseline_start,
                "baseline_stop": baseline_stop,
                "baseline_mean": float(mean_value),
                "baseline_std": float(std_value),
            }
        )
    return pd.DataFrame.from_records(rows, columns=BASELINE_SUMMARY_COLUMNS)
