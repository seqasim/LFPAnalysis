# Legacy Function-to-New API Mapping

## Mapping table

| Old entry point | First migration target | Notes |
| --- | --- | --- |
| `make_mne(...)` | `load_lfp(...)` or `build_basic_pipeline_config(...)` + `run_pipeline(...)` | Use `LFPAnalysis.legacy.make_mne` if you need a bridge |
| `ref_mne(...)` | `preprocess_lfp(...)` or `ReferenceConfig(...)` inside `run_pipeline(...)` | Stable path for `none`, `wm`, and `bipolar` |
| `make_epochs(...)` | `make_epochs(raw, EpochConfig(...))` or `build_event_locked_pipeline_config(...)` | Stable path covers the no-side-effects case |
| `compute_and_baseline_tfr(...)` | Save task + baseline `-epo.fif` files (chapter 07), then `load_lfp` + `run_analysis(..., baseline_epochs=...)` with `TfrConfig` crop on the TFR axis | Cross-event trialwise Morlet is stable; full legacy orchestration remains advanced |
| `compute_connectivity(...)` | advanced utilities + migration guide | Still advanced, not fully wrapped |

## Side-by-side example

#### Old workflow

```python
epochs = lfp_preprocess_utils.make_epochs(
    load_path=f"{load_path}/sample_ieeg_bp.fif",
    slope=slope,
    offset=offset,
    behav_name="feedback_start",
    behav_times=behav_data["feedback_start"].tolist(),
    ev_start_s=0.5,
    ev_end_s=1.5,
)
```

#### New workflow

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
```

## What changed conceptually

The new path asks you to distinguish between event timestamps, baseline choices, and later analyses instead of bundling them into one utility call.

Next step: {doc}`22_translate_condensed_notebook`
