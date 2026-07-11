"""Root conftest loaded before pytest-cov starts measuring imports.

Importing NumPy here prevents a known NumPy/pandas reload breakage when
coverage begins tracing ``LFPAnalysis`` imports that pull in pandas.
"""

from __future__ import annotations

import numpy as np  # noqa: F401
