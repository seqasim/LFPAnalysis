"""
Basic tests for nlx_utils module.
"""
import pytest
from LFPAnalysis import nlx_utils


def test_module_imports():
    """Test that nlx_utils module can be imported."""
    assert hasattr(nlx_utils, '__name__')
    assert nlx_utils.__name__ == 'LFPAnalysis.nlx_utils'

