"""
Basic tests for analysis_utils module.
"""
import pytest
import numpy as np
import pandas as pd
from LFPAnalysis import analysis_utils


def test_module_imports():
    """Test that analysis_utils module can be imported and has expected structure."""
    # Just verify the module exists and can be accessed
    assert hasattr(analysis_utils, '__name__')
    assert analysis_utils.__name__ == 'LFPAnalysis.analysis_utils'

