"""Unit tests for baseline helpers in lfp_preprocess_utils."""

from __future__ import annotations

import numpy as np
import pytest

lfp_preprocess_utils = pytest.importorskip("LFPAnalysis.lfp_preprocess_utils")


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["mean", "ratio", "percent", "zscore", "logratio", "zlogratio"])
def test_mean_baseline_time_supported_modes(mode):
    data = np.abs(np.random.randn(2, 100)) + 0.1
    baseline = np.abs(np.random.randn(2, 50)) + 0.1
    result = lfp_preprocess_utils.mean_baseline_time(data, baseline, mode=mode)
    assert result.shape == data.shape
    assert isinstance(result, np.ndarray)


@pytest.mark.unit
def test_baseline_avg_tfr_returns_expected_shape():
    data = np.abs(np.random.randn(2, 4, 20)) + 0.1
    baseline = np.abs(np.random.randn(2, 4, 10)) + 0.1
    result = lfp_preprocess_utils.baseline_avg_TFR(data, baseline, mode="zscore")
    assert result.shape == data.shape
