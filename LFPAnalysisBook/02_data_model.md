# What Files You Need Before Starting

## What this step is for

This chapter explains the repository's input contract so you know which files must exist before a workflow can succeed.

## When you should use it

Read this before trying to reference channels, epoch events, or run analysis notebooks on your own dataset.

## Required inputs

- electrophysiology data in EDF, Neuralynx, or MNE FIF form
- optional electrode metadata in CSV or XLSX form
- optional behavioral event timestamps in seconds

## Minimal example

```python
from LFPAnalysis import load_electrode_metadata

electrodes = load_electrode_metadata("../tests/data/electrodes.csv")
print(electrodes.columns)
```

## How to inspect the result

Confirm that your table includes `label` and that channel names match the electrophysiology file or can be translated deterministically.

## Common mistakes

- mixing milliseconds and seconds in behavioral timestamps
- assuming the package performs electrode localization itself
- treating atlas columns as required when only `label` is required for validation

## Old-to-new translation note

The old notebooks often encoded file assumptions inline. The refactored repo makes the file contract explicit and validates it before running the rest of the workflow.

## Supported inputs and outputs

- inputs: EDF, Neuralynx `.ncs`, MNE FIF, electrode CSV/XLSX, behavior timestamps
- outputs: MNE `Raw`, MNE `Epochs`, artifact tables, baseline summaries, spectral outputs

Next step: {doc}`03_first_load`
