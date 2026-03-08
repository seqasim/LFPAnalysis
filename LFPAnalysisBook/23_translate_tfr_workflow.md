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
from pathlib import Path
from LFPAnalysis import build_event_locked_pipeline_config, run_pipeline

config = build_event_locked_pipeline_config(
    Path("../data/sample_ieeg_continuous_rest.fif"),
    file_format="mne",
    event_name="feedback_start",
    event_times=[5.0, 10.0, 15.0],
)
result = run_pipeline(config)
epochs = result.epochs
# continue with MNE time-frequency functions or the advanced utilities
```

## What changed conceptually

The refactored repo separates preparation from time-frequency computation. You prepare trusted epochs first, then run TFR code as an explicit second stage.

## Where behavior is not identical

A fully wrapped stable replacement for `compute_and_baseline_tfr(...)` does not exist yet. Use the compatibility shim or the advanced utilities when you need the original behavior.

Next step: {doc}`24_translate_connectivity_workflow`
