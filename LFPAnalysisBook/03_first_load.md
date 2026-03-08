# Your First Successful Load

## What this step is for

This is the first end-to-end success checkpoint. You should be able to point the stable API at bundled sample data and inspect the returned MNE object.

## When you should use it

Use this before making any preprocessing decisions. If loading is unclear, every downstream choice becomes harder.

## Required inputs

- a supported file path
- the file format label

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(Path("../data/sample_ieeg.fif"), file_format="mne")
result = run_pipeline(config)
```

## How to inspect the result

Check:

- `result.raw.info["sfreq"]`
- `result.raw.ch_names[:10]`
- `result.metadata`

## Common mistakes

- using a directory path when the stable API expects a specific file path
- choosing the wrong `file_format`
- trying to reference channels before checking the loaded data object

## Old-to-new translation note

If you previously started by manually reading files or calling `make_mne`, the new recommended first step is a builder plus `run_pipeline`.

## Worked example and smoke checks

Read the worked notebook first, then use the smoke notebooks when you only want to verify that the environment still works.

Next step: {doc}`04_first_reference`
