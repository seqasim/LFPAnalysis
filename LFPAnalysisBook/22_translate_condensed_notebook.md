# Translating the Condensed Notebook

## Old condensed pattern

The condensed notebook usually did three things early:

1. create or load an MNE object
2. re-reference it
3. epoch it around behavior timestamps

## Side-by-side translation

#### Old workflow

```python
mne_data = lfp_preprocess_utils.make_mne(load_path=load_path, elec_path=elec_path, format="edf")
mne_data_reref = lfp_preprocess_utils.ref_mne(mne_data=mne_data, elec_path=elec_path, method="bipolar")
epochs = lfp_preprocess_utils.make_epochs(
    load_path=f"{load_path}/sample_ieeg_bp.fif",
    slope=slope,
    offset=offset,
    behav_name="feedback_start",
    behav_times=behav_data["feedback_start"].tolist(),
    ev_start_s=0.5,
    ev_end_s=1.5,
)
```

#### New workflow

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
    tmin=-0.5,
    tmax=1.5,
    reference_method="none",
    baseline_mode="zscore",
    baseline_window=(-0.5, 0.0),
    metadata={"reward": beh["reward"].tolist(), "rpe": beh["rpe"].tolist()},
)
result = run_pipeline(config)
```

## Where behavior is not identical

- the stable path does not recreate every legacy side effect such as writing artifact CSVs to disk
- the stable path makes baseline explicit instead of hiding it inside a broader epoching flow
- connectivity remains outside the stable pipeline; beginner Morlet TFR is available via `run_analysis`

Next step: {doc}`23_translate_tfr_workflow`
