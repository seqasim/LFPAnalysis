"""
Basic import tests to verify all modules can be imported successfully.
"""
import pytest


def test_import_analysis_utils():
    """Test that analysis_utils can be imported."""
    import LFPAnalysis.analysis_utils
    assert LFPAnalysis.analysis_utils is not None


def test_import_lfp_preprocess_utils():
    """Test that lfp_preprocess_utils can be imported."""
    import LFPAnalysis.lfp_preprocess_utils
    assert LFPAnalysis.lfp_preprocess_utils is not None


def test_import_oscillation_utils():
    """Test that oscillation_utils can be imported."""
    import LFPAnalysis.oscillation_utils
    assert LFPAnalysis.oscillation_utils is not None


def test_import_statistics_utils():
    """Test that statistics_utils can be imported."""
    import LFPAnalysis.statistics_utils
    assert LFPAnalysis.statistics_utils is not None


def test_import_sync_utils():
    """Test that sync_utils can be imported."""
    import LFPAnalysis.sync_utils
    assert LFPAnalysis.sync_utils is not None


def test_import_nlx_utils():
    """Test that nlx_utils can be imported."""
    import LFPAnalysis.nlx_utils
    assert LFPAnalysis.nlx_utils is not None


def test_import_iowa_utils():
    """Test that iowa_utils can be imported."""
    import LFPAnalysis.iowa_utils
    assert LFPAnalysis.iowa_utils is not None

