# What Remains Legacy-Only or Advanced

## Still advanced or legacy-oriented

- full legacy TFR orchestration (`compute_and_baseline_tfr`); beginner Morlet TFR is available via `TfrConfig` / `run_analysis`
- low-level behavioral sync helpers (`sync_utils.synchronize_data*`) — typed prep uses `SyncConfig` / `run_prep` instead
- connectivity orchestration across the many legacy options
- unimplemented reference methods such as `laplacian` (omitted from the stable registry)
- notebook-specific side effects such as writing intermediate artifact CSVs to disk
- soft-archived stubs under `LFPAnalysis._scratch_utils` (still importable from original modules with deprecation warnings)

## How to choose responsibly

- use the stable API first for preparation
- use `LFPAnalysis.legacy` if you are actively translating old code
- drop to advanced utilities only after you know why you need them

## Side-by-side example

#### Old workflow

```python
lfp_preprocess_utils.compute_and_baseline_tfr(...)
```

#### New workflow

```python
import numpy as np
from LFPAnalysis import build_analysis_config, run_analysis

# Beginner Morlet on existing Epochs:
result = run_analysis(
    epochs,
    build_analysis_config(tfr_method="morlet", tfr_freqs=np.arange(4, 30, 4).tolist()),
)

# Full legacy orchestration when you need the old helper surface:
from LFPAnalysis import legacy
legacy.compute_and_baseline_tfr(...)
```
## Why this is explicit now

The refactored repo aims to reduce ambiguity. It is better to label a workflow as advanced or compatibility-only than to imply that the stable API already covers it.

Next step: {doc}`30_troubleshooting`
