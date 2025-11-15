"""
Basic tests for statistics_utils module.
"""
import pytest
from LFPAnalysis import statistics_utils


def test_module_imports():
    """Test that statistics_utils module can be imported."""
    assert hasattr(statistics_utils, '__name__')
    assert statistics_utils.__name__ == 'LFPAnalysis.statistics_utils'

