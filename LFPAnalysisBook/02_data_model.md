# The Bundled Gambling-Task Dataset

## What this step is for

This chapter introduces the sample dataset that drives the entire beginner track. Every later chapter loads these same files so you learn one coherent analysis story—not disconnected toy examples.

## When you should use it

Read this before loading data, referencing channels, synchronizing behavior, or epoching events. When you move to your own lab data, mirror this file layout.

## Required inputs

The bundled files under `../data/`:

| File | Role |
|------|------|
| `sample_ieeg.fif` | 22-channel monopolar sEEG, 500 Hz, ~788 s |
| `sample_ieeg_bp.fif` | 15 bipolar pairs derived from the same recording |
| `sample_photodiode.fif` | Photodiode sync trace, 1024 Hz |
| `sample_beh.csv` | 80 trials: event times + behavior |
| `sample_ts.csv` | 486 behavioral sync pulses (superset of trial events) |
| `sample_labels.xlsx` | Monopolar electrode metadata (coordinates, atlas) |
| `sample_labels_bp` | Bipolar pair metadata (`salman_region`, `hemisphere`) |
| `sample_feedback_start-epo.fif` | 80 feedback-locked epochs (pre-built) |
| `sample_baseline_start-epo.fif` | 80 baseline-locked epochs (pre-built) |
| `feedback_start_artifact_df.csv` | Misc-artifact times binned by trial/channel |
| `baseline_start_IED_df.csv` | IED sidecar (empty in this sample) |

### Behavioral columns (`sample_beh.csv`)

All times are in **seconds** (not milliseconds):

- `feedback_start` — feedback screen onset (primary epoching anchor)
- `baseline_start` — pre-gamble baseline screen onset
- `reward` — 1 = win, 0 = loss
- `gamble_rt` — reaction time to gamble choice (seconds)
- `rpe` — reward prediction error (continuous)

### `sample_beh.csv` vs `sample_ts.csv`

- **`sample_beh.csv`** (80 rows): one row per trial with summary behavior. Use this for epoching and condition contrasts.
- **`sample_ts.csv`** (486 rows): every sync pulse from the task computer. Use this with the photodiode to recover the linear map between behavioral and neural clocks.

All 160 trial event times (`feedback_start` + `baseline_start`) appear inside `sample_ts`, but `sample_ts` also contains pulses outside the neural recording window.

## Minimal example

```python
import pandas as pd
from pathlib import Path

beh = pd.read_csv(Path("../data/sample_beh.csv"))
print(f"Trials: {len(beh)}")
print(beh[["feedback_start", "reward", "rpe"]].head())
print(f"Feedback times span {beh.feedback_start.min():.1f}–{beh.feedback_start.max():.1f} s")
```

## How to inspect the result

Confirm:

- 80 trials with fractional-second timestamps
- `reward` is 0/1 and `rpe` is continuous
- Event times fall inside the ~788 s neural recording (all trial times in this sample do)

## Common mistakes

- Treating `sample_ts.csv` as a per-trial table (it is a sync-pulse log, not one row per trial)
- Mixing milliseconds and seconds when epoching
- Assuming the package performs electrode localization (it consumes existing tables)
- Expecting `sample_labels.xlsx` to have a `label` column out of the box (it ships `NMMlabel`; see chapter 04)

## Old-to-new translation note

The old Condensed Notebook encoded these file assumptions inline. The refactored book makes the contract explicit: neural FIF + behavior CSV + electrode metadata + optional photodiode for sync.

## Supported inputs and outputs

- **Inputs:** MNE FIF, electrode CSV/XLSX, behavioral CSV, photodiode FIF
- **Outputs:** MNE `Raw`, MNE `Epochs` (with metadata), artifact tables, baseline summaries, spectral/TFR/connectivity results

Next step: {doc}`03_first_load`
