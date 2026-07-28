# Translating the TFR Workflow

### Old workflow

```python
lfp_preprocess_utils.compute_and_baseline_tfr(
    baseline_event,
    task_events,
    freqs,
    n_cycles,
    load_path,
    save_path,
)
```

### New workflow

```python
import pandas as pd
import numpy as np
from pathlib import Path
from LFPAnalysis import build_analysis_config, build_event_locked_pipeline_config, run_analysis, run_pipeline

beh = pd.read_csv(Path("../data/sample_beh.csv"))
prep = run_pipeline(
    build_event_locked_pipeline_config(
        Path("../data/sample_ieeg_bp.fif"),
        file_format="mne",
        event_name="feedback_start",
        event_times=beh["feedback_start"].tolist(),
        tmin=-0.5,
        tmax=1.5,
        baseline_mode="none",
    )
)
freqs = np.arange(4, 30, 4).tolist()
tfr = run_analysis(
    prep.epochs.copy().pick(["racas1-racas2"]),
    build_analysis_config(tfr_method="morlet", tfr_freqs=freqs, tfr_n_cycles=3.0),
)
print(tfr.tfr["power"].data.shape)
```

## What changed conceptually

The refactored repo separates preparation from time-frequency computation. You prepare trusted epochs first, then run beginner Morlet TFR on the analysis spine.

## Where behavior is not identical

Beginner Morlet TFR is available via `TfrConfig` / `run_analysis`. A full drop-in replacement for every option of `compute_and_baseline_tfr(...)` (multi-event orchestration, disk layout) does not exist yet — use the compatibility shim or advanced utilities when you need that exact legacy behavior.

Next step: {doc}`24_translate_connectivity_workflow`
