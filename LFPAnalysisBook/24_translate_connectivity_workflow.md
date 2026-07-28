# Translating the Connectivity Workflow

### Old workflow

```python
pwise = oscillation_utils.compute_connectivity(
    epochs_reref.copy(),
    band=band,
    metric=metric,
    indices=indices,
)
```

### New workflow

```python
import pandas as pd
from pathlib import Path
from LFPAnalysis import build_event_locked_pipeline_config, run_pipeline

beh = pd.read_csv(Path("../data/sample_beh.csv"))
config = build_event_locked_pipeline_config(
    Path("../data/sample_ieeg_bp.fif"),
    file_format="mne",
    event_name="feedback_start",
    event_times=beh["feedback_start"].tolist(),
    tmin=-0.5,
    tmax=1.5,
)
result = run_pipeline(config)
epochs = result.epochs
# continue with oscillation_utils.compute_connectivity(...) or mne-connectivity
```

## What changed conceptually

The stable API prepares the data object and records preprocessing decisions. The actual connectivity metric remains an advanced step that you run explicitly.

## Where behavior is not identical

Connectivity is still not fully wrapped in the stable API. That is intentional and documented instead of hidden.

Next step: {doc}`25_legacy_only_surfaces`
