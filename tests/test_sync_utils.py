"""Unit tests for synchronization utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from LFPAnalysis import sync_utils


@pytest.mark.unit
def test_moving_average_returns_expected_length():
    data = np.arange(10)
    result = sync_utils.moving_average(data, 3)
    assert len(result) == 8
    assert np.isclose(result[0], 1.0)


@pytest.mark.unit
def test_moving_average_window_of_one_returns_original_values():
    data = np.array([1, 2, 3, 4])
    result = sync_utils.moving_average(data, 1)
    np.testing.assert_array_equal(result, data)


@pytest.mark.unit
def test_get_neural_ts_ttl_reads_positive_timestamps():
    nev_data = {"records": {"ttl": np.array([0, 1, 0, 2]), "TimeStamp": np.array([0, 1_000_000, 2_000_000, 3_000_000])}}
    neural_ts = sync_utils.get_neural_ts_ttl(nev_data)
    np.testing.assert_allclose(neural_ts, np.array([1.0]))
