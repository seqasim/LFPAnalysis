# Your First Baseline

## What this step is for

Choose and inspect a baseline correction on real feedback-locked epochs. Baselining expresses power or amplitude relative to a pre-event window—a core interpretability step before comparing conditions.

## When you should use it

Use this after you have Epochs (here: the bundled `sample_feedback_start-epo.fif`) and before comparing reward vs no-reward responses. Chapter 07 shows the same baselining choice when creating epochs from continuous data.

## Required inputs

- Epoched data (`../data/sample_feedback_start-epo.fif` or epochs you create in chapter 07)
- Baseline mode (e.g. `zscore`)
- Baseline window overlapping the pre-stimulus period

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_analysis_config, load_lfp, run_analysis
from LFPAnalysis.config import LoadConfig

epochs = load_lfp(
    LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne", preload=True)
)
result = run_analysis(
    epochs,
    build_analysis_config(baseline_mode="zscore", baseline_window=(-0.5, 0.0)),
)
print(result.baseline_summary.head())
print(result.epochs.times[[0, -1]])
```

Use `build_analysis_config` + `run_analysis` for baseline-only teaching on existing Epochs. Do **not** use `build_spectral_pipeline_config` here — that helper always enables PSD/FOOOF.

For event-locked baselining from continuous data, use `build_event_locked_pipeline_config` with real `feedback_start` times from `sample_beh.csv` (chapter 07). When epoching is enabled, baselined Epochs land in `PipelineResult.epochs`. When you only load continuous data and enable analysis without epoching, the analysis object is stored in `PipelineResult.referenced`.

## How to inspect the result

Review `result.baseline_summary`:

- `baseline_mean` and `baseline_std` per channel
- Confirm the window `(-0.5, 0.0)` overlaps `result.epochs.times`
- Confirm `result.spectral` is empty for a baseline-only run
- The worked notebook ({doc}`worked-examples/06_first_baseline_run`) plots baseline-corrected evoked activity for one channel

## Common mistakes

- Enabling baselining without setting a window
- Using a baseline mode by habit instead of because it fits the design (`zscore` is common for oscillatory power)
- Assuming the baseline summary is optional bookkeeping—it is QA
- Using `build_spectral_pipeline_config` for a baseline-only demo (it also runs PSD)

## Old-to-new translation note

Older notebooks mixed epoch creation and baselining in one flow. The refactored API makes the baseline choice explicit on the analysis spine and returns a dedicated summary table.

Next step: {doc}`07_first_event_locked_workflow`
