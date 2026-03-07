"""Unit tests for validation helpers and metadata loading."""

from __future__ import annotations

import pandas as pd
import pytest

from LFPAnalysis.exceptions import ConfigurationError, DataContractError
from LFPAnalysis.validation import ensure_supported, validate_required_columns
from LFPAnalysis.workflow import load_electrode_metadata


@pytest.mark.unit
def test_ensure_supported_accepts_known_value():
    assert ensure_supported("mne", field_name="file_format", supported=("mne", "edf")) == "mne"


@pytest.mark.unit
def test_ensure_supported_rejects_unknown_value():
    with pytest.raises(ConfigurationError):
        ensure_supported("csv", field_name="file_format", supported=("mne", "edf"))


@pytest.mark.unit
def test_validate_required_columns_rejects_missing_columns():
    dataframe = pd.DataFrame({"channel": ["l1"]})
    with pytest.raises(DataContractError):
        validate_required_columns(dataframe, required_columns=("label",), schema_name="Electrodes")


@pytest.mark.unit
def test_load_electrode_metadata_reads_csv(electrode_csv_path):
    dataframe = load_electrode_metadata(electrode_csv_path)
    assert list(dataframe.columns[:4]) == ["label", "x", "y", "z"]
    assert dataframe["label"].tolist() == ["l1", "l2", "r1"]
