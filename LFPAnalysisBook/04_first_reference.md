# Your First Reference Choice

## What this step is for

Apply a reference scheme to the real monopolar recording using the bundled electrode table, then compare bipolar output to the pre-built `sample_ieeg_bp.fif`. Referencing changes amplitude interpretation, so this chapter makes those choices explicit.

## When you should use it

Use this after loading `sample_ieeg.fif` and before spectral or event-locked analysis.

## Required inputs

- `../data/sample_ieeg.fif` — monopolar raw
- `../data/sample_labels.xlsx` — electrode metadata (coordinates, atlas labels)
- `../data/sample_ieeg_bp.fif` — pre-referenced bipolar version for comparison

### Column-name gotcha (generalizes to your own data)

`load_electrode_metadata` now accepts either `label` or `NMMlabel` and normalizes to `label`. This keeps the bundled `sample_labels.xlsx` compatible with the stable API:

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
# After bipolar re-reference, prep drops the superseded monopolar Raw to save RAM.
print(f"Bipolar: {len(result.referenced.ch_names)} ch")
```

White-matter referencing (`wm`) additionally requires a `manual` column; the sample table uses `Manual Examination` instead—another rename you would need for `wm`.

## Reference options in the stable path

- `none`: no re-reference
- `bipolar`: adjacent contact differences (`a-b`)
- `wm`: each gray-matter contact minus a selected white-matter contact
- `car`: common average reference across good channels
- `car_trimmed`: common average reference using a trimmed mean (default trims top 20% and bottom 20% at each time sample before averaging)

Use `car_trimmed` when you expect occasional high-amplitude outlier channels that would otherwise pull the common average.

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

Inspect `result.referenced.ch_names` (15 bipolar pairs like `racas1-racas2`). After a true re-reference, `result.raw` is intentionally `None` (the superseded monopolar object is dropped to save RAM). Load `sample_ieeg_bp.fif` separately and confirm channel names match. The worked notebook ({doc}`worked-examples/04_first_preprocessing_run`) plots one bipolar channel and demonstrates `car_trimmed`.

## The re-referenced electrode sheet

When an electrode sheet is provided (`electrode_path`), `result.electrode_df` now follows the reference method:

- `bipolar`: one row per virtual pair (`a-b`), with midpoint averages for coordinate columns (`x/y/z`, `mni_x/mni_y/mni_z`) and provenance columns `anode`, `cathode`
- `wm`: one row per pair, inheriting metadata/coordinates from the anode (gray-matter contact)
- `none`, `car`, `car_trimmed`: channel names are unchanged, so the original electrode sheet is returned unchanged

Example save step:

```python
from pathlib import Path

out = Path("results/demo/reference")
out.mkdir(parents=True, exist_ok=True)
result.electrode_df.to_csv(out / "sample_labels_bp.csv", index=False)
```

## Common mistakes

- Choosing `wm` or `bipolar` without a valid electrode table
- Assuming `laplacian` is available in the stable path (it is not registered)
- Expecting `result.raw` to remain available after bipolar/wm re-reference (prep drops it)
- Forgetting to record which reference was used in your analysis notes
- Using plain `car` when a few outlier channels dominate the average (prefer `car_trimmed`)

## Old-to-new translation note

The old repo called `ref_mne` directly. The stable path wraps the same decision in `ReferenceConfig` or the convenience builders.

## Not yet supported in the stable path

`laplacian` is intentionally omitted from the stable reference registry until implemented.

## Legacy parity note

The stable path now preserves the legacy ordering for first-pass prep decisions:

- load-stage conditioning first (clinical notch + 500 Hz resample defaults)
- reference after load conditioning
- electrode sheet outputs matched to the selected reference policy (`bipolar`/`wm` derive new labels; `car`/`car_trimmed` keep original labels)

Next step: {doc}`04b_first_synchronization`
