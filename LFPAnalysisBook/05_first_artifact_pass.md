# Your First Artifact Pass

## What this step is for

Run a first-pass artifact screen on the real bipolar recording and relate detector output to the bundled sidecar CSVs. Artifact tables are quality-control signals, not ground truth.

## When you should use it

Use this on continuous data before event-locked averaging or spectral interpretation.

## Required inputs

- `../data/sample_ieeg_bp.fif` — bipolar continuous recording
- Detector choice: `misc`, `ied`, or both

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(
    Path("../data/sample_ieeg_bp.fif"),
    file_format="mne",
    artifact_methods=["misc", "ied"],
)
result = run_pipeline(config)

misc_table = result.artifact_tables["misc"]
ied_table = result.artifact_tables["ied"]
print(f"Misc events: {len(misc_table)}, IED events: {len(ied_table)}")
```

Compare with the sidecar files `feedback_start_artifact_df.csv` (pre-binned by trial for epoched workflows).

## How to inspect the result

- Row count per detector
- Which channels are flagged most often
- Whether timestamps cluster in one recording segment
- The worked notebook plots flagged time ranges on a short raw segment

## Common mistakes

- Treating detector output as ground truth instead of QC
- Adding `ied` by default when the task does not require it (this sample's IED sidecars are empty)
- Expecting the stable API to write artifact CSVs to disk (legacy `make_epochs` did; stable API returns tables in memory)

## Old-to-new translation note

Old notebooks called `detect_misc_artifacts` or `detect_IEDs` directly. The stable path wraps those into standardized tables with columns `event_kind`, `channel`, `time_seconds`, `sample_index`.

Next step: {doc}`06_first_baseline`
