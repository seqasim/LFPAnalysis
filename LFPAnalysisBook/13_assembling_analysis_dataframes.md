# Assembling Analysis DataFrames

## What this step is for

Every function in `statistics_utils` expects a **tidy long dataframe**: one row per observation, with columns for the neural value, behavioral regressors, electrode identity, participant, trial, and (for time-resolved work) a time index `ts`. This chapter shows how to build that table from MNE epochs, behavioral metadata, and ROI labels — the glue between spectral metrics and group statistics.

## When you should use it

Use this after you have event-locked epochs (or band-power / TFR estimates) and ROI assignments (chapter 12). Do it before calling `permutation_regression_zscore` or `time_resolved_mlm`.

## Required inputs

- Epochs with trial-aligned metadata (e.g. `rpe`, `reward`) — `../data/sample_feedback_start-epo.fif` + `../data/sample_beh.csv`
- Electrode table with `label` and ROI columns — `../data/sample_labels_bp`
- A chosen channel (or set of channels) and a scalar or time-resolved neural measure

## Minimal example

Single-patient, single-channel, time-resolved band power → long dataframe:

```python
import numpy as np
import pandas as pd
import mne
from pathlib import Path

beh = pd.read_csv(Path("../data/sample_beh.csv"))
epochs = mne.read_epochs(Path("../data/sample_feedback_start-epo.fif"), preload=True, verbose=False)
epochs.metadata = beh[["reward", "rpe"]].copy()
epochs.metadata["trial"] = np.arange(len(epochs))

elec_df = pd.read_csv(Path("../data/sample_labels_bp"))
chan = "racas1-racas2"
roi = elec_df.loc[elec_df.label == chan, "salman_region"].iloc[0]

# Beta-band envelope power at subsampled time points
ep = epochs.copy().pick([chan]).filter(13, 30, verbose=False)
times = ep.times
step = max(1, len(times) // 20)
rows = []
for t_idx in range(0, len(times), step):
    power = ep.get_data()[:, 0, t_idx] ** 2
    for trial_i, p in enumerate(power):
        rows.append(
            {
                "participant": "sample",
                "unique_label": chan,
                "roi": roi,
                "trial": trial_i,
                "ts": times[t_idx],
                "tfr": p,
                "rpe": epochs.metadata["rpe"].iloc[trial_i],
                "reward": epochs.metadata["reward"].iloc[trial_i],
            }
        )
smoothed_df = pd.DataFrame(rows)
print(smoothed_df.head())
print(smoothed_df.shape)
```

Column conventions expected downstream:

| Column | Role |
| --- | --- |
| `tfr` (or your `y`) | Dependent variable |
| `ts` | Time point for time-resolved models |
| `unique_label` | Lower grouping unit (usually electrode / bipolar pair) |
| `participant` | Higher grouping unit (patient) |
| `trial` | Trial id within participant (needed for hierarchical shuffling) |
| regressors | e.g. `rpe`, `reward`, interactions |

### Multi-patient shape

The sample dataset is one patient. When you concatenate several patients, keep the same columns and ensure `participant` differs; electrode labels that collide across patients should be made unique (common pattern: `f"{participant}_{chan}"` stored in `unique_label`).

```python
# Conceptual: after repeating the block above per subject
# all_df = pd.concat(per_subject_dfs, ignore_index=True)
```

Chapter 14 builds a small **synthetic** multi-patient dataframe inline so you can run mixed-effects models without additional sample files.

## How to inspect the result

- One row per (participant × electrode × trial × time) for time-resolved data
- No NaNs in `y`, grouping columns, or regressors you will put in the formula
- `ts` unique values match the time axis you intend to plot
- `unique_label` is constant within an electrode’s rows; `participant` is constant within a patient

## Common mistakes

- Feeding wide arrays (trials × times) directly into `statistics_utils` — melt/reshape first
- Forgetting `trial` when you plan to use `time_resolved_mlm` / `shuffle_data_for_mlm`
- Reusing the same `unique_label` across patients when concatenating
- Aligning metadata by position after dropping epochs — re-check lengths after any epoch rejection

## Old-to-new translation note

`TimeResolvedRegression.ipynb` and similar legacy notebooks often built this table ad hoc. The column names above (`unique_label`, `participant`, `trial`, `ts`) match what `shuffle_data_for_mlm` and `time_resolved_mlm` default to today.

Next step: {doc}`14_group_level_statistics`
