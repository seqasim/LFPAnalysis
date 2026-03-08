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


@pytest.mark.unit
def test_normalized_sliding_windows_returns_unit_norm_rows():
    windows = sync_utils._normalized_sliding_windows(np.array([1.0, 3.0, 6.0, 10.0]), window_size=2)
    np.testing.assert_allclose(windows.mean(axis=1), np.zeros(windows.shape[0]))
    np.testing.assert_allclose(np.linalg.norm(windows, axis=1), np.ones(windows.shape[0]))


@pytest.mark.unit
def test_pulsealign_matches_nonuniform_pulse_sequences():
    beh_ts = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
    neural_ts = beh_ts + 0.5

    good_beh, good_neural = sync_utils.pulsealign(beh_ts, neural_ts, windSize=3)

    assert len(good_beh) >= 3
    np.testing.assert_allclose(good_neural - good_beh, np.full(len(good_beh), 0.5))


@pytest.mark.unit
def test_synchronize_data_robust_recovers_linear_offset():
    beh_ts = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
    neural_ts = beh_ts + 0.5

    slope, offset, rval = sync_utils.synchronize_data_robust(
        beh_ts=beh_ts,
        neural_ts=neural_ts,
        window_size=3,
        step_size=1,
        correlation_threshold=0.99,
    )

    assert slope == pytest.approx(1.0)
    assert offset == pytest.approx(0.5)
    assert rval == pytest.approx(1.0)
