# Your First Event-Locked Workflow

## What this step is for

Build **feedback-locked epochs** from real trial times in `sample_beh.csv`, attach `reward`, `rpe`, and `gamble_rt` as epoch metadata, and apply baseline correction. Baselining expresses amplitude relative to a reference window—either a pre-event period on the same epoch, or a window locked to a *different* per-trial event.

This enables condition contrasts (`reward == 1` vs `reward == 0`) for every downstream chapter.

## When you should use it

Use this when you have behavioral timestamps and want a clean event-locked object before PSD, TFR, connectivity, or statistics.

## Required inputs

- `../data/sample_ieeg_bp.fif` — bipolar continuous (times already in neural seconds)
- `../data/sample_beh.csv` — `feedback_start`, `baseline_start`, `reward`, `rpe`, `gamble_rt`

## Minimal example (same-event baseline)

By default, `baseline_window` is relative to the epoch event (`feedback_start` here):

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
    baseline_mode="zscore",
    baseline_window=(-0.5, 0.0),
    tmin=-0.5,
    tmax=1.5,
    metadata={
        "reward": beh["reward"].tolist(),
        "rpe": beh["rpe"].tolist(),
        "gamble_rt": beh["gamble_rt"].tolist(),
    },
)
result = run_pipeline(config)

reward_epochs = result.epochs[result.epochs.metadata["reward"] == 1]
loss_epochs = result.epochs[result.epochs.metadata["reward"] == 0]
print(f"Reward trials: {len(reward_epochs)}, Loss trials: {len(loss_epochs)}")
print(result.baseline_summary.head())
```

`zscore` is common for oscillatory / amplitude work. Other modes (`mean`, `ratio`, `percent`, `logratio`, `zlogratio`) are available when the design calls for them. Review `result.baseline_summary` (`baseline_mean`, `baseline_std` per channel) as QA—it is not optional bookkeeping.

## Cross-event baselining

Sometimes the baseline should not be a window on the task epoch itself. For example, you may want to lock analysis epochs to `recog_time` but take the baseline from a window around `baseline_time_mem` on each trial.

Pass `baseline_event_times` (same length as `event_times`). Then `baseline_window` is interpreted **relative to each baseline event**, not relative to the task event:

```python
config = build_event_locked_pipeline_config(
    Path("../data/sample_ieeg_bp.fif"),
    file_format="mne",
    event_name="feedback_start",
    event_times=beh["feedback_start"].tolist(),
    baseline_mode="zscore",
    # Per-trial baseline locked to a different event column:
    baseline_event_times=beh["baseline_start"].tolist(),
    baseline_window=(-0.5, 0.0),  # relative to baseline_start
    tmin=-0.5,
    tmax=1.5,
    metadata={"reward": beh["reward"].tolist(), "rpe": beh["rpe"].tolist()},
)
result = run_pipeline(config)
```

Prep extracts a second set of epochs around the baseline events; analysis applies per-trial mean/std from those windows to the task epochs. This is the pattern for designs like baselining `recog_time` to `baseline_time_mem`.

## Baseline-only on existing Epochs

If you already have Epochs on disk and only need to baseline them (no re-epoching), use the analysis spine:

```python
from LFPAnalysis import build_analysis_config, load_lfp, run_analysis
from LFPAnalysis.config import LoadConfig

epochs = load_lfp(
    LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne", preload=True)
)
analysis = run_analysis(
    epochs,
    build_analysis_config(baseline_mode="zscore", baseline_window=(-0.5, 0.0)),
)
```

Do **not** use `build_spectral_pipeline_config` for a baseline-only demo—that helper always enables PSD/FOOOF.

The worked notebook ({doc}`worked-examples/07_first_epoching_run`) plots the evoked average for reward vs no-reward and demonstrates cross-event baselining with `baseline_start`.

## How to inspect the result

- `len(result.epochs)` — expect 80
- `result.epochs.times[[0, -1]]` — pre/post window
- `result.epochs.metadata[["reward", "rpe"]].describe()`
- `result.baseline_summary.head()` — confirm mode and window
- For cross-event runs, `result.metadata["cross_event_baseline"]` / `has_baseline_epochs` should be true

## Common mistakes

- Passing milliseconds instead of seconds
- Using hardcoded demo times `[5.0, 10.0, 15.0]` instead of real `feedback_start` values
- Forgetting `metadata` dict keys must have the same length as `event_times`
- Enabling baselining without setting a window
- Assuming same-event baselining when you need a different event stream (use `baseline_event_times`)
- Passing `baseline_event_times` with a different length than `event_times`
- Expecting the stable API to write artifact CSV sidecars to disk

## Old-to-new translation note

This is the stable replacement for the old `make_epochs(...)` pattern when you need epochs with behavioral metadata and flexible baselining, not every legacy side effect. Older notebooks mixed epoch creation and baselining in one opaque flow; the refactored API makes the baseline choice (same-event vs cross-event) explicit.

Next step: {doc}`08_first_psd_and_fooof`
