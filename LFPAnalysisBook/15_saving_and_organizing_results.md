# Saving and Organizing Results

## What this step is for

Spectral and statistical workflows produce many intermediate objects (epochs, TFRs, FOOOF tables, regression results). This chapter documents **what the library already saves**, what stays in memory, and a simple on-disk layout for group-level tidy tables so collaborators can reload analyses without re-running everything.

## When you should use it

Use this whenever you leave the interactive case study and start accumulating per-electrode or per-patient outputs — before plotting summaries (chapter 16) or sharing results.

## Required inputs

- Paths you control for outputs (avoid hard-coded cluster paths from examples)
- Awareness of which helper you called (`compute_and_baseline_tfr`, `compute_FOOOF_parallel`, `run_pipeline`, or your own dataframe export)

## What exists today

### In-memory pipeline container

`PipelineResult` (`LFPAnalysis.results`) holds `raw`, `referenced`, `epochs`, `artifact_tables`, `baseline_summary`, `spectral`, `tfr`, `electrode_df`, `sync`, and `metadata`. It does **not** write to disk. Persist anything you need explicitly (e.g. `epochs.save(...)`, `DataFrame.to_csv` / `to_parquet`).

`electrode_df` reflects your reference method:

- `bipolar`: virtual pair labels (`a-b`) with midpoint coordinate averages and `anode`/`cathode` provenance
- `wm`: pair labels with anode-inherited coordinates/metadata
- `none`, `car`, `car_trimmed`: unchanged original contact sheet

### TFR helpers

`lfp_preprocess_utils.compute_and_baseline_tfr(..., save_path=..., output='save')` writes MNE TFR HDF5 files as `{save_path}/{event}-tfr.h5`. Related array-based helpers can write `.npz` when `output` is `'save'` or `'both'`.

Artifact tables from epoching are written as CSVs under a bads path (`{behav_name}_IED_df.csv`, `{behav_name}_artifact_df.csv`).

### FOOOF parallel helper

`analysis_utils.compute_FOOOF_parallel(..., do_save=True, save_path=...)` writes:

- `{save_path}/{subj_id}/scratch/FOOOF/{event_name}/dfs/{chan_name}_df.csv`
- optional plots under `.../plots/` when `do_plot=True`

The default `save_path` in the signature is a lab cluster path — **always override it** for local work.

## Recommended layout for group analyses

Documentation-only convention (not enforced by code):

```text
results/
  <study>/
    subjects/
      <participant>/
        epochs/
        tfr/
        fooof/
        connectivity/
    group/
      tidy/
        bandpower_long.parquet   # chapter 13 shape
        mlm_time_resolved.csv    # chapter 14 outputs
      figures/
```

Minimal persistence for stats outputs:

```python
from pathlib import Path

out = Path("results/demo/group/tidy")
out.mkdir(parents=True, exist_ok=True)
smoothed_df.to_parquet(out / "bandpower_long.parquet", index=False)
# or: smoothed_df.to_csv(out / "bandpower_long.csv", index=False)
mlm_res.to_csv(out / "mlm_time_resolved.csv", index=False)
# if available from run_pipeline(...):
# result.electrode_df.to_csv(out / "electrodes_referenced.csv", index=False)
```

Prefer parquet for large long tables; CSV is fine for small regression summaries and for readers who do not have `pyarrow`/`fastparquet`.

## How to inspect the result

- After TFR save: open with `mne.time_frequency.read_tfrs` and check channel count / times
- After FOOOF save: reload the per-channel CSV and confirm `participant`, `region`, `cond`, `frequency` columns
- After group export: `pd.read_*` and verify dtypes for `participant`, `unique_label`, `trial`, `ts`

## Common mistakes

- Relying on `PipelineResult` as an archive — it disappears when the process ends
- Leaving the default FOOOF `save_path` pointing at another machine’s filesystem
- Saving only wide matrices without the tidy metadata needed to re-run statistics
- Mixing absolute lab paths into notebooks you commit to the repo

## Old-to-new translation note

Legacy notebooks often scattered `.npy` / `.csv` writes beside each analysis cell. Prefer a single study-level `results/` tree and keep the **long dataframe** as the source of truth for anything that will enter `statistics_utils`.

Next step: {doc}`16_plotting_recipes`
