"""
Basic tests for iowa_utils module.
"""
import pytest
from LFPAnalysis import iowa_utils


def test_module_imports():
    """Test that iowa_utils module can be imported."""
    assert hasattr(iowa_utils, '__name__')
    assert iowa_utils.__name__ == 'LFPAnalysis.iowa_utils'

