# Your First Event-Locked Workflow

## What this step is for

Build **feedback-locked raw epochs** from real trial times in `sample_beh.csv`, attach `reward`, `rpe`, and `gamble_rt` metadata, and optionally stage cross-event baseline epochs for TFR in chapter 09.

## When you should use it

Use this when you have behavioral timestamps and want a clean event-locked object before PSD, TFR, connectivity, or statistics.

## Required inputs

- `../data/sample_ieeg_bp.fif` — bipolar continuous (times already in neural seconds)
- `../data/sample_beh.csv` — `feedback_start`, `baseline_start`, `reward`, `rpe`, `gamble_rt`

## Minimal example (raw epochs, no voltage baseline)

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
    baseline_mode="none",
    tmin=-0.5,
    tmax=1.5,
    metadata={
        "reward": beh["reward"].tolist(),
        "rpe": beh["rpe"].tolist(),
        "gamble_rt": beh["gamble_rt"].tolist(),
    },
)
result = run_pipeline(config)
print(f"epochs: {len(result.epochs)}")
print("baseline summary rows:", len(result.baseline_summary))
```

Epochs remain raw by design. Baseline normalization now belongs to the TFR stage, not to voltage epochs.

## Cross-event baseline epochs for TFR

For legacy-style cross-event normalization, pass `baseline_event_times` plus a `baseline_window`. Prep will extract and carry baseline-event epochs (`result.metadata["has_baseline_epochs"] == True`) so chapter 09 can apply trialwise TFR z-scoring.

```python
cross_cfg = build_event_locked_pipeline_config(
    Path("../data/sample_ieeg_bp.fif"),
    file_format="mne",
    event_name="feedback_start",
    event_times=beh["feedback_start"].tolist(),
    baseline_mode="trialwise",
    baseline_event_times=beh["baseline_start"].tolist(),
    baseline_window=(-0.5, 0.0),  # relative to baseline_start
    tmin=-0.5,
    tmax=1.5,
    metadata={"reward": beh["reward"].tolist(), "rpe": beh["rpe"].tolist()},
)
cross = run_pipeline(cross_cfg)
print(cross.metadata["cross_event_baseline"], cross.metadata["has_baseline_epochs"])
```

## How to inspect the result

- `len(result.epochs)` — expect 80
- `result.epochs.times[[0, -1]]` — pre/post window
- `result.epochs.metadata[["reward", "rpe"]].describe()`
- `len(result.baseline_summary)` — expect `0` for raw-epoch runs
- For cross-event runs, `result.metadata["cross_event_baseline"]` / `has_baseline_epochs` should be true

## Common mistakes

- Passing milliseconds instead of seconds
- Using hardcoded demo times `[5.0, 10.0, 15.0]` instead of real `feedback_start` values
- Forgetting `metadata` keys must match `event_times` length
- Expecting voltage-domain baselining in this chapter (removed by design)
- Passing `baseline_event_times` with a different length than `event_times`
- Expecting the stable API to write artifact CSV sidecars to disk

## Old-to-new translation note

This replaces the old `make_epochs(...)` pattern for creating event-locked epochs with metadata. Legacy-style normalization now occurs in chapter 09 at the TFR stage.

Next step: {doc}`08_first_psd_and_fooof`
