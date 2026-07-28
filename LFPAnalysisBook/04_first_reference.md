# Your First Reference Choice

## What this step is for

Apply **bipolar referencing** to the real monopolar recording using the bundled electrode table, then compare your result to the pre-built `sample_ieeg_bp.fif`. Referencing changes amplitude interpretation—this chapter makes that choice explicit.

## When you should use it

Use this after loading `sample_ieeg.fif` and before spectral or event-locked analysis.

## Required inputs

- `../data/sample_ieeg.fif` — monopolar raw
- `../data/sample_labels.xlsx` — electrode metadata (coordinates, atlas labels)
- `../data/sample_ieeg_bp.fif` — pre-referenced bipolar version for comparison

### Column-name gotcha (generalizes to your own data)

The stable helper `load_electrode_metadata` requires a column named `label`. The bundled `sample_labels.xlsx` ships `NMMlabel` instead. The legacy `load_elec` helper renames `NMMlabel` → `label` automatically, which is why `preprocess_lfp` works when you pass the xlsx path directly:

```python
from pathlib import Path
import pandas as pd
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

# Stable validation path — rename first if you use load_electrode_metadata:
elec = pd.read_excel(Path("../data/sample_labels.xlsx"))
elec = elec.rename(columns={"NMMlabel": "label"})

config = build_basic_pipeline_config(
    Path("../data/sample_ieeg.fif"),
    file_format="mne",
    reference_method="bipolar",
    electrode_path=Path("../data/sample_labels.xlsx"),  # legacy load_elec handles NMMlabel
)
result = run_pipeline(config)
# After bipolar re-reference, prep drops the superseded monopolar Raw to save RAM.
print(f"Bipolar: {len(result.referenced.ch_names)} ch")
```

White-matter referencing (`wm`) additionally requires a `manual` column; the sample table uses `Manual Examination` instead—another rename you would need for `wm`.

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(
    Path("../data/sample_ieeg.fif"),
    file_format="mne",
    reference_method="bipolar",
    electrode_path=Path("../data/sample_labels.xlsx"),
)
result = run_pipeline(config)
```

## How to inspect the result

Inspect `result.referenced.ch_names` (15 bipolar pairs like `racas1-racas2`). After a true re-reference, `result.raw` is intentionally `None` (the superseded monopolar object is dropped to save RAM). Load `sample_ieeg_bp.fif` separately and confirm channel names match. The worked notebook ({doc}`worked-examples/04_first_preprocessing_run`) plots one bipolar channel.

## Common mistakes

- Choosing `wm` or `bipolar` without a valid electrode table
- Assuming `laplacian` is available in the stable path (it is not registered; use bipolar/wm or advanced utilities)
- Expecting `result.raw` to remain available after bipolar/wm re-reference (prep drops it)
- Forgetting to record which reference was used in your analysis notes
- Passing `sample_labels.xlsx` to `load_electrode_metadata` without renaming `NMMlabel`

## Old-to-new translation note

The old repo called `ref_mne` directly. The stable path wraps the same decision in `ReferenceConfig` or the convenience builders.

## Not yet supported in the stable path

`laplacian` is intentionally omitted from the stable reference registry until implemented.

Next step: {doc}`04b_first_synchronization`
