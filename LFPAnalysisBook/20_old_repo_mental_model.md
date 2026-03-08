# Coming From The Old Repo: Mental Model

## What changed conceptually

The old repo centered on utility modules and notebooks that mixed data loading, side effects, and analysis code in one place.

The refactored repo separates three layers:

- `stable beginner-facing API`
- `compatibility/legacy shims`
- `advanced legacy utility modules`

## What stayed the same

- MNE remains the core analysis object model
- the legacy utility modules still exist
- the old algorithms are still present even when the stable API does not yet wrap them

## What changed in day-to-day use

- you now choose an interface first
- configs or builders define workflow intent explicitly
- compatibility wrappers tell you the new equivalent instead of silently leaving you in the old path

## Side-by-side example

#### Old workflow

```python
mne_data = lfp_preprocess_utils.make_mne(load_path=load_path, elec_path=elec_path, format="edf")
mne_data_reref = lfp_preprocess_utils.ref_mne(mne_data=mne_data, elec_path=elec_path, method="bipolar")
```

#### New workflow

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(
    Path("../data/sample_ieeg.fif"),
    file_format="mne",
    reference_method="none",
)
result = run_pipeline(config)
```

## What changed conceptually in that example

The new path makes configuration explicit and returns a structured result object. The old path emphasized direct utility calls.

Next step: {doc}`21_legacy_function_mapping`
