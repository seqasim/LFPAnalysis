"""Unit tests for workflow-layer behavior that does not require full analysis extras."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from LFPAnalysis import ArtifactConfig, BaselineConfig, detect_artifacts
from LFPAnalysis.schemas import ARTIFACT_EVENT_COLUMNS
from LFPAnalysis.workflow import baseline_lfp


@pytest.mark.unit
def test_detect_artifacts_accepts_custom_mapping():
    dummy = SimpleNamespace(info={"sfreq": 100.0})
    result = detect_artifacts(
        dummy,
        ArtifactConfig(methods=["custom"], custom_detector=lambda data: {"l1": [0.1, 0.3]}),
    )
    assert list(result["custom"].columns) == list(ARTIFACT_EVENT_COLUMNS)
    assert len(result["custom"]) == 2


@pytest.mark.unit
def test_detect_artifacts_accepts_custom_dataframe():
    dummy = SimpleNamespace(info={"sfreq": 100.0})
    dataframe = pd.DataFrame({"event_kind": ["custom"], "channel": ["l1"], "time_seconds": [0.1], "sample_index": [10]})
    result = detect_artifacts(
        dummy,
        ArtifactConfig(methods=["custom"], custom_detector=lambda data: dataframe),
    )
    assert result["custom"].equals(dataframe)


@pytest.mark.unit
def test_baseline_lfp_noop_when_disabled():
    data = object()
    returned, summary = baseline_lfp(data, BaselineConfig(enabled=False))
    assert returned is data
    assert summary.empty
