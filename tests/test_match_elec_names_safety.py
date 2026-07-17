"""Safety tests for match_elec_names non-interactive behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from LFPAnalysis.analysis_utils import FOOOF_continuous
from LFPAnalysis.lfp_preprocess_utils import match_elec_names


def test_match_elec_names_raises_on_ambiguous_tie_noninteractive():
    """Ambiguous Levenshtein ties must raise ValueError (not call input())."""
    mne_names = ["raa1", "rbb1"]
    loc_names = pd.Series(["raa1", "rxx1"])  # rxx1 unmatched; ties both mne names

    with pytest.raises(ValueError, match="Ambiguous electrode match"):
        match_elec_names(mne_names, loc_names, interactive=False)


def test_match_elec_names_happy_path_exact_overlap():
    mne_names = ["ra1", "ra2"]
    loc_names = pd.Series(["ra1", "ra2"])
    new_names, unmatched_names, unmatched_seeg = match_elec_names(
        mne_names, loc_names, interactive=False
    )
    assert new_names == ["ra1", "ra2"]
    assert unmatched_seeg == []


def test_archived_stub_raises_not_implemented():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(NotImplementedError):
            FOOOF_continuous([1.0, 2.0])
