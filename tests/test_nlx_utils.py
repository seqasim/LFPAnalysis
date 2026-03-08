"""Unit tests for Neuralynx utility helpers."""

from __future__ import annotations

import datetime as dt

import numpy as np
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


@pytest.mark.unit
def test_parse_subject_nlx_data_skips_dropped_channels_before_loading(monkeypatch):
    load_calls = []

    def fake_load_ncs(path):
        load_calls.append(path)
        return {"data": np.array([1.0]), "sampling_rate": 1000}

    monkeypatch.setattr(nlx_utils, "load_ncs", fake_load_ncs)

    signals, srs, names, types = nlx_utils.parse_subject_nlx_data(
        ["subject/LFPx1.ncs", "subject/LFPx2.ncs"],
        seeg_names=["lfpx2"],
        drop_names=["LFPx1"],
    )

    assert load_calls == ["subject/LFPx2.ncs"]
    assert len(signals) == 1
    assert srs == [1000]
    assert names == ["lfpx2"]
    assert types == ["seeg"]


@pytest.mark.unit
def test_parse_subject_nlx_data_assigns_single_channel_type(monkeypatch):
    monkeypatch.setattr(
        nlx_utils,
        "load_ncs",
        lambda path: {"data": np.array([1.0, 2.0]), "sampling_rate": 2000},
    )

    _, _, names, types = nlx_utils.parse_subject_nlx_data(
        ["subject/EKG1.ncs", "subject/LFPx3_extra.ncs"],
        seeg_names=["lfpx3"],
    )

    assert names == ["ekg1", "lfpx3"]
    assert types == ["ecg", "seeg"]
