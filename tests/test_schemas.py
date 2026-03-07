"""Unit tests for standard workflow output schemas."""

from __future__ import annotations

import numpy as np
import pytest

from LFPAnalysis.schemas import build_baseline_summary, build_event_table


@pytest.mark.unit
def test_build_event_table_creates_long_form_output():
    table = build_event_table({"l1": np.array([0.1, 0.2]), "l2": np.array([0.3])}, event_kind="misc", sfreq=100)
    assert list(table.columns) == ["event_kind", "channel", "time_seconds", "sample_index"]
    assert len(table) == 3
    assert set(table["channel"]) == {"l1", "l2"}


@pytest.mark.unit
def test_build_baseline_summary_tracks_per_channel_statistics():
    table = build_baseline_summary(
        target="raw",
        channel_names=["l1", "l2"],
        mode="zscore",
        baseline_start=-0.5,
        baseline_stop=0.0,
        baseline_mean=np.array([1.0, 2.0]),
        baseline_std=np.array([0.1, 0.2]),
    )
    assert table.shape == (2, 7)
    assert table.loc[0, "mode"] == "zscore"
