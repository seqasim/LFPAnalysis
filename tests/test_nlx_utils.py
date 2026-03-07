"""Unit tests for Neuralynx utility helpers."""

from __future__ import annotations

import datetime as dt

import pytest

from LFPAnalysis import nlx_utils


@pytest.mark.unit
def test_parse_neuralynx_time_string_returns_datetime():
    parsed = nlx_utils.parse_neuralynx_time_string("-TimeCreated By DataAcqSystem At 01/02/2024 03:04:05.123")
    assert isinstance(parsed, dt.datetime)
    assert parsed.year == 2024
    assert parsed.minute == 4


@pytest.mark.unit
def test_estimate_record_count_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        nlx_utils.estimate_record_count(tmp_path / "missing.ncs", nlx_utils.NCS_RECORD)
