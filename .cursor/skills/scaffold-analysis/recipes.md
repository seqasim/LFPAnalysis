# Recipe Menu

Select the minimal set of recipe IDs that cover the Dataset Profile goals. Read canonical sources before writing each notebook; adapt to the user's PATHS keys and regressors.

## Selection rules

1. Prefer the smallest set that satisfies stated goals.
2. Include `preprocess_to_epochs` only when stage is `raw` or `referenced_raw`.
3. If stage is `tfr`, skip `tfr_baselined`; start from band averaging / dataframe / stats.
4. For multi-ROI or multi-subject stats, include `dataframe_assembly` before or inside the regression notebooks.
5. Number notebooks so a linear workflow reads in order (`00_`, `01_`, …).

## Default bundles

**Preprocessed re-referenced epochs + behavior → TFR, band power, regressions**

| Order | Recipe ID | Notebook |
|------|-----------|----------|
| 1 | `tfr_baselined` | `01_baselined_tfr.ipynb` |
| 2 | `band_power` | `02_band_power.ipynb` |
| 3 | `regression_avg` | `03_time_averaged_regression.ipynb` |
| 4 | `regression_timeresolved` | `04_time_resolved_regression.ipynb` |

Optional: `dataframe_assembly` as `02b_assemble_analysis_df.ipynb` when building multi-channel / multi-subject long tables.

**Spectral parameterization only:** `psd_fooof`

**Connectivity:** `connectivity` (optionally after epochs exist)

---

## Recipes

### `preprocess_to_epochs`

| Field | Value |
|-------|-------|
| Notebook | `00_preprocess_to_epochs.ipynb` |
| When | stage is raw or referenced continuous |
| API | Stable: `build_*_pipeline_config`, `run_pipeline` / staged `load_lfp` → `preprocess_lfp` → `detect_artifacts` → `make_epochs` |
| Sources | `LFPAnalysisBook/03_first_load.md` … `07_first_event_locked_workflow.md`; `worked-examples/03`–`07` |

Scaffold: LoadConfig/paths, reference method, artifact methods, event times from behavior, save epochs path placeholder.

### `tfr_baselined`

| Field | Value |
|-------|-------|
| Notebook | `01_baselined_tfr.ipynb` |
| When | User wants Morlet (or similar) TFR and/or condition contrasts |
| API | Load epochs via `load_lfp` or `mne.read_epochs`; beginner Morlet via `build_analysis_config(..., tfr_method="morlet")` + `run_analysis`; advanced: `epochs.compute_tfr` / `lfp_preprocess_utils.compute_and_baseline_tfr` |
| Sources | `LFPAnalysisBook/09_first_time_frequency.md`, `LFPAnalysisBook/worked-examples/09_first_tfr_run.ipynb`, `LFPAnalysisBook/23_translate_tfr_workflow.md` |

Scaffold: attach behavior metadata, pick channel(s), frequency grid, optional condition-split TFR, plot difference map, optional save of TFR.

### `band_power`

| Field | Value |
|-------|-------|
| Notebook | `02_band_power.ipynb` |
| When | Extract band-limited power (time-averaged and/or time-resolved) for stats |
| API | Filter + power on epochs, or average TFR over frequency (and optionally time); build trial-level arrays/tables |
| Sources | `LFPAnalysisBook/10b_first_time_resolved_stats.md`, `LFPAnalysisBook/13_assembling_analysis_dataframes.md` |

Scaffold: band edges (e.g. beta 13–30), channel pick, time-averaged power per trial, subsampled time-resolved power with `ts` column when feeding time-resolved recipes.

### `regression_avg`

| Field | Value |
|-------|-------|
| Notebook | `03_time_averaged_regression.ipynb` |
| When | Regress trial-level (time-averaged) power against behavior |
| API | `LFPAnalysis.statistics_utils.permutation_regression_zscore` |
| Sources | `LFPAnalysisBook/10b_first_time_resolved_stats.md`, `LFPAnalysisBook/worked-examples/10b_first_stats_run.ipynb`, `LFPAnalysisBook/14_group_level_statistics.md` |

Scaffold: build `model_df` with `power` + regressors; formula e.g. `power ~ rpe`; set `np.random.seed` before permutations; note permutation count for dry-run vs publication.

### `regression_timeresolved`

| Field | Value |
|-------|-------|
| Notebook | `04_time_resolved_regression.ipynb` |
| When | Effects across time (`ts`) — per-electrode loops or multi-subject MLM |
| API | Loop `permutation_regression_zscore` over time for single-subject; `statistics_utils.time_resolved_mlm` for multi-subject long dataframes |
| Sources | `LFPAnalysisBook/14_group_level_statistics.md`, `LFPAnalysisBook/worked-examples/14_group_statistics.ipynb`, `LFPAnalysisBook/13_assembling_analysis_dataframes.md` |

Scaffold: require long dataframe columns `participant`, `unique_label`, `trial`, `ts`, `tfr` (or `y`), regressors; single-subject path uses electrode-wise permutation OLS over time; document that MLM needs multiple participants.

### `psd_fooof`

| Field | Value |
|-------|-------|
| Notebook | `01_psd_fooof.ipynb` |
| When | PSD and/or FOOOF parameterization |
| API | Stable spectral config via pipeline when appropriate; else analysis utilities / book patterns |
| Sources | `LFPAnalysisBook/08_first_psd_and_fooof.md`, `LFPAnalysisBook/worked-examples/08_first_psd_and_fooof_run.ipynb` |

### `dataframe_assembly`

| Field | Value |
|-------|-------|
| Notebook | `02_assemble_analysis_df.ipynb` (or `02b_…` if band_power is `02_`) |
| When | Building tidy long tables for stats across channels/subjects |
| API | pandas + epoch metadata + electrode ROI join; column conventions for `statistics_utils` |
| Sources | `LFPAnalysisBook/13_assembling_analysis_dataframes.md`, `LFPAnalysisBook/worked-examples/13_assembling_dataframes.ipynb` |

Required columns for downstream stats: `participant`, `unique_label`, `trial`, `ts` (if time-resolved), neural `y` (`tfr` or `power`), regressors.

### `connectivity`

| Field | Value |
|-------|-------|
| Notebook | `01_connectivity.ipynb` |
| When | Connectivity and surrogate baselines |
| API | `oscillation_utils` connectivity helpers + `make_surrogate_data` |
| Sources | `LFPAnalysisBook/10_first_connectivity_and_surrogates.md`, `LFPAnalysisBook/worked-examples/10_first_connectivity_run.ipynb` |

---

## API layer reminder

| Step | Layer |
|------|-------|
| Load, reference, artifacts, epochs, baseline prep | Stable (`config`, `workflow`, builders) |
| TFR, connectivity, PAC, custom band-power glue | Advanced utilities / MNE |
| Permutation OLS, time-resolved MLM | `statistics_utils` |
