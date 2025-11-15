"""
Basic tests for sync_utils module.
"""
import pytest
import numpy as np
from LFPAnalysis import sync_utils


def test_moving_average():
    """Test moving_average function with default window size."""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    n = 3
    
    result = sync_utils.moving_average(a, n)
    
    # Result should be shorter by n-1
    assert len(result) == len(a) - n + 1
    assert isinstance(result, np.ndarray)
    # First value should be average of first n values
    assert np.isclose(result[0], np.mean(a[:n]))


def test_moving_average_default_window():
    """Test moving_average function with default window size."""
    a = np.random.randn(20)
    
    result = sync_utils.moving_average(a)
    
    # Default window is 11, so result should be len(a) - 10
    assert len(result) == len(a) - 10
    assert isinstance(result, np.ndarray)


def test_moving_average_single_value():
    """Test moving_average with window size of 1."""
    a = np.array([1, 2, 3, 4, 5])
    n = 1
    
    result = sync_utils.moving_average(a, n)
    
    # With window of 1, should return same values
    assert len(result) == len(a)
    np.testing.assert_array_almost_equal(result, a)

