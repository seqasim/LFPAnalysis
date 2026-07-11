"""Pytest configuration and shared fixtures for LFPAnalysis tests."""

from __future__ import annotations

from pathlib import Path

# NumPy must remain importable before fixtures build MNE objects. The repo-root
# conftest.py preloads NumPy for pytest-cov; keep this import for fixture use.
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPO_ROOT / "tests" / "data"


@pytest.fixture
def electrode_csv_path() -> Path:
    return TEST_DATA / "electrodes.csv"


@pytest.fixture
def mne_module():
    return pytest.importorskip("mne")


@pytest.fixture
def synthetic_raw(mne_module):
    sfreq = 200.0
    times = np.arange(0, 20, 1 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 8 * times),
            np.cos(2 * np.pi * 12 * times),
        ]
    )
    info = mne_module.create_info(["l1", "l2"], sfreq, ch_types=["seeg", "seeg"])
    return mne_module.io.RawArray(data, info, verbose=False)


@pytest.fixture
def synthetic_epochs(mne_module, synthetic_raw):
    events = np.array([[400, 0, 1], [1200, 0, 1], [2000, 0, 1]])
    return mne_module.Epochs(
        synthetic_raw.copy(),
        events=events,
        event_id={"demo": 1},
        tmin=-0.5,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
