"""
Basic tests for oscillation_utils module.
"""
import pytest
import numpy as np
from LFPAnalysis import oscillation_utils


def test_find_nearest_value():
    """Test find_nearest_value function."""
    array = np.array([1.0, 2.5, 3.7, 5.2, 6.8])
    value = 3.0
    
    nearest_val, idx = oscillation_utils.find_nearest_value(array, value)
    
    # Should find 2.5 or 3.7 as nearest
    assert nearest_val in array
    assert idx >= 0
    assert idx < len(array)
    assert nearest_val == array[idx]


def test_find_nearest_value_exact_match():
    """Test find_nearest_value with exact match."""
    array = np.array([1.0, 2.5, 3.7, 5.2, 6.8])
    value = 3.7
    
    nearest_val, idx = oscillation_utils.find_nearest_value(array, value)
    
    assert nearest_val == 3.7
    assert idx == 2


def test_find_nearest_value_at_boundary():
    """Test find_nearest_value at array boundaries."""
    array = np.array([1.0, 2.5, 3.7, 5.2, 6.8])
    
    # Test value smaller than min
    nearest_val, idx = oscillation_utils.find_nearest_value(array, 0.5)
    assert idx == 0
    
    # Test value larger than max
    nearest_val, idx = oscillation_utils.find_nearest_value(array, 10.0)
    assert idx == len(array) - 1

