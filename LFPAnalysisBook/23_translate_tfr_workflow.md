# Translating the TFR Workflow

### Old workflow

```python
lfp_preprocess_utils.compute_and_baseline_tfr(
    baseline_event,
    task_events,
    freqs,
    n_cycles,
    load_path,
    save_path,
)
```

### New workflow (cross-event, two saved epoch files)

```python
import numpy as np
from pathlib import Path
from LFPAnalysis import load_lfp, run_analysis
from LFPAnalysis.config import AnalysisConfig, LoadConfig, TfrConfig

task_epochs = load_lfp(
    LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne", preload=True)
)
baseline_epochs = load_lfp(
    LoadConfig(path=Path("../data/sample_baseline_start-epo.fif"), file_format="mne", preload=True)
)
freqs = np.arange(4, 30, 4).tolist()

result = run_analysis(
    task_epochs,
    AnalysisConfig(
        tfr=TfrConfig(
            enabled=True,
            method="morlet",
            freqs=freqs,
            n_cycles=3.0,
            baseline_mode="trialwise",
            apply_baseline=True,
            crop_tmin=-0.5,
            crop_tmax=1.5,
            baseline_crop_tmin=-0.5,
            baseline_crop_tmax=0.0,
        )
    ),
    baseline_epochs=baseline_epochs,
)
print(result.tfr["power"].data.shape)
print(result.tfr["power"].times[0], result.tfr["power"].times[-1])
```

Voltage epochs keep their prep buffers for Morlet. `crop_tmin` / `crop_tmax` trim the **TFR** time axis after Morlet; use `baseline_crop_*` when the baseline core window differs from the task window.

### Simple Morlet (single epoch file, no cross-event baseline)

```python
import numpy as np
from pathlib import Path
from LFPAnalysis import build_analysis_config, load_lfp, run_analysis
from LFPAnalysis.config import LoadConfig

epochs = load_lfp(
    LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne", preload=True)
)
freqs = np.arange(4, 30, 4).tolist()
tfr = run_analysis(
    epochs.copy().pick(["racas1-racas2"]),
    build_analysis_config(tfr_method="morlet", tfr_freqs=freqs, tfr_n_cycles=3.0),
)
print(tfr.tfr["power"].data.shape)
```

## What changed conceptually

The refactored repo separates preparation from time-frequency computation. Save trusted raw epoch files once (chapter 07), then run TFR many times via `run_analysis` without re-epoching from continuous data.

## Where behavior is not identical

Beginner Morlet TFR is available via `TfrConfig` / `run_analysis`. A full drop-in replacement for every option of `compute_and_baseline_tfr(...)` (multi-event orchestration, disk layout) does not exist yet — use the compatibility shim or advanced utilities when you need that exact legacy behavior.

Next step: {doc}`24_translate_connectivity_workflow`
