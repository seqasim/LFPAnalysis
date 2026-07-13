# Anatomy and ROI Assignment

## What this step is for

After you have per-electrode metrics (power, FOOOF peaks, connectivity, etc.), you almost always need to ask: *which brain region is this electrode in?* This chapter shows how to map electrode metadata to custom ROI labels using the packaged YBA atlas helpers, and how to build seed/target channel sets for pairwise analyses.

## When you should use it

Use this before group-level statistics, ROI-averaged summaries, or seed-based connectivity. It sits after the single-patient case study (chapters 02–11) and before assembling tidy analysis dataframes (chapter 13).

## Required inputs

- Bipolar electrode metadata with atlas columns — for the sample data use `../data/sample_labels_bp` (CSV). Key columns: `label`, `NMM`, `BN246`, `YBA_1`, plus `salman_region` and `hemisphere` for connectivity seed/target helpers.
- Packaged atlas table `data/YBA_ROI_labelled.xlsx` (loaded automatically by `analysis_utils`).
- For `make_seed_target_df`: an MNE `Epochs` object whose channel names match `elec_df.label`.

## Minimal example

```python
import pandas as pd
from pathlib import Path
from LFPAnalysis import analysis_utils, oscillation_utils
from LFPAnalysis import load_lfp
from LFPAnalysis.config import LoadConfig

elec_df = pd.read_csv(Path("../data/sample_labels_bp"))
# select_rois_picks expects a Manual/collapsed column name; sample file uses ManualExamination
chan = "racas1-racas2"
roi = analysis_utils.select_rois_picks(elec_df, chan, manual_col="ManualExamination")
print(chan, "->", roi)

# Reverse lookup: which electrodes fall in a named anatomical string?
acc_picks = analysis_utils.select_picks_rois(elec_df, roi="cingulate")
print("cingulate picks:", acc_picks[:5])

# Seed/target pairs for connectivity (uses salman_region + hemisphere)
epochs = load_lfp(LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne"))
seed_target = oscillation_utils.make_seed_target_df(
    elec_df, epochs, source_roi="ACC", target_roi="OFC"
)
print(seed_target)
```

`select_rois_picks` resolves a single channel to a custom ROI by cascading through YBA → manual label → BN246 → NMM heuristics (for example mapping entorhinal / hippocampus / amygdala when atlas columns disagree). `select_picks_rois` goes the other direction: given an ROI string or list, return matching electrode labels. `make_seed_target_df` builds hemisphere-wise seed and target channel *indices* for connectivity metrics.

## How to inspect the result

- For each channel of interest, print the assigned ROI and compare it to `YBA_1` / `NMM` / `ManualExamination` in the table.
- Confirm bipolar `label` strings match epoch channel names exactly (sample bipolar names are lowercase with hyphens, e.g. `racas1-racas2`).
- For seed/target tables: both `seed` and `target` lists should be non-empty in at least one hemisphere; empty hemispheres are dropped.

## Common mistakes

- Passing monopolar labels (`RaCaS1`) when epochs are bipolar (`racas1-racas2`) — use `sample_labels_bp`, not `sample_labels.xlsx`, with bipolar epochs.
- Forgetting `manual_col` when your electrode table does not have `collapsed_manual` (the default argument).
- Assuming `salman_region` is produced by `select_rois_picks` — in the sample file it is already a column; for new patients you typically assign it yourself (often by applying `select_rois_picks` across all labels).
- Mixing hemisphere casing (`r` vs `R`); `make_seed_target_df` lowercases hemisphere before matching.

## Old-to-new translation note

Older notebooks often hard-coded ROI string filters on `YBA_1` / `NMM`. Prefer `select_rois_picks` / `select_picks_rois` so atlas fallbacks stay consistent, then feed `salman_region` into `make_seed_target_df` for connectivity.

Next step: {doc}`13_assembling_analysis_dataframes`
