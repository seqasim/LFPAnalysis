"""
Basic tests for lfp_preprocess_utils module.
"""
import pytest
import numpy as np
from LFPAnalysis import lfp_preprocess_utils


def test_mean_baseline_time_zscore():
    """Test mean_baseline_time function with zscore mode."""
    # Create simple test data
    data = np.random.randn(2, 100)  # 2 channels, 100 time points
    baseline = np.random.randn(2, 50)  # 2 channels, 50 baseline time points
    
    result = lfp_preprocess_utils.mean_baseline_time(data, baseline, mode='zscore')
    
    # Check that output has correct shape
    assert result.shape == data.shape
    # Check that it's a numpy array
    assert isinstance(result, np.ndarray)


def test_mean_baseline_time_mean():
    """Test mean_baseline_time function with mean mode."""
    data = np.random.randn(2, 100)
    baseline = np.random.randn(2, 50)
    
    result = lfp_preprocess_utils.mean_baseline_time(data, baseline, mode='mean')
    
    assert result.shape == data.shape
    assert isinstance(result, np.ndarray)


def test_mean_baseline_time_ratio():
    """Test mean_baseline_time function with ratio mode."""
    data = np.abs(np.random.randn(2, 100)) + 0.1  # Ensure positive values
    baseline = np.abs(np.random.randn(2, 50)) + 0.1
    
    result = lfp_preprocess_utils.mean_baseline_time(data, baseline, mode='ratio')
    
    assert result.shape == data.shape
    assert isinstance(result, np.ndarray)

