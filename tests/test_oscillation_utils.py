"""Tests for oscillation utility helpers that require optional analysis dependencies."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mne_connectivity")
oscillation_utils = pytest.importorskip("LFPAnalysis.oscillation_utils")


@pytest.mark.unit
@pytest.mark.optional_dep
def test_find_nearest_value_exact_match():
    array = np.array([1.0, 2.5, 3.7, 5.2])
    nearest_val, idx = oscillation_utils.find_nearest_value(array, 3.7)
    assert nearest_val == 3.7
    assert idx == 2


@pytest.mark.unit
@pytest.mark.optional_dep
def test_make_surrogate_arrays_is_deterministic():
    data = np.arange(12, dtype=float).reshape(3, 4)
    first = list(oscillation_utils.make_surrogate_arrays(data, n_shuffles=2, rng_seed=11, return_generator=False))
    second = list(oscillation_utils.make_surrogate_arrays(data, n_shuffles=2, rng_seed=11, return_generator=False))
    assert len(first) == 2
    assert np.array_equal(first[0], second[0])
