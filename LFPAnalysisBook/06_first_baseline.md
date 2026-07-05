# Your First Baseline

## What this step is for

Choose and inspect a baseline correction on real feedback-locked epochs. Baselining expresses power or amplitude relative to a pre-event window—a core interpretability step before comparing conditions.

## When you should use it

Use this after epoching and before comparing reward vs no-reward responses.

## Required inputs

- Epoched data (we use pre-built `sample_feedback_start-epo.fif` or epochs you create in chapter 07)
- Baseline mode (e.g. `zscore`)
- Baseline window overlapping the pre-stimulus period

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_spectral_pipeline_config, run_pipeline

config = build_spectral_pipeline_config(
    Path("../data/sample_feedback_start-epo.fif"),
    file_format="mne",
    baseline_mode="zscore",
    baseline_window=(-0.5, 0.0),
)
result = run_pipeline(config)
print(result.baseline_summary.head())
```

For event-locked baselining from continuous data, use `build_event_locked_pipeline_config` with real `feedback_start` times from `sample_beh.csv` (chapter 07).

## How to inspect the result

Review `result.baseline_summary`:

- `baseline_mean` and `baseline_std` per channel
- Confirm the window `(-0.5, 0.0)` overlaps `epochs.times`
- The worked notebook plots baseline-corrected evoked activity for one channel

## Common mistakes

- Enabling baselining without setting a window
- Using a baseline mode by habit instead of because it fits the design (`zscore` is common for oscillatory power)
- Assuming the baseline summary is optional bookkeeping—it is QA

## Old-to-new translation note

Older notebooks mixed epoch creation and baselining in one flow. The refactored API makes the baseline choice explicit and returns a dedicated summary table.

Next step: {doc}`07_first_event_locked_workflow`
