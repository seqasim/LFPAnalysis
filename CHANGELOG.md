# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Dual-spine stable API: `PrepConfig` / `run_prep` / `PrepResult` (raw → Epochs, owns sync + electrodes) and `AnalysisConfig` / `run_analysis` / `AnalysisResult` (starts from Epochs).
- `SyncConfig`, `ElectrodeConfig`, `TfrConfig`; schemas for sync summary and TFR metadata.
- Builders: `build_prep_config`, `build_analysis_config`, `build_tutorial_pipeline_config`.
- `LFPAnalysis.advanced` package with lazy curated exports for post-handoff science.
- `run_pipeline` is now a tutorial convenience that runs prep then analysis.
- Modern packaging metadata and optional dependency groups.
- Contributor and governance documentation for open-source collaboration.
- Explicit `joblib` dependency for optional parallel surrogate / stats paths.
- `LFPAnalysis._scratch_utils` soft-archive for stubs / unused helpers (import via original module names with `DeprecationWarning`).

### Changed
- `ReferenceMethod` / reference registry no longer list unimplemented `laplacian` (was a false-support trap).
- `EpochConfig` gains `buffer_s` for prep-time epoch padding.
- Default working dtype for signal and TFR arrays is now `float32` (`WORKING_DTYPE`) to roughly halve RAM on local machines.
- `LoadConfig.preload` now defaults to `False` (lazy / disk-backed loads). Set `preload=True` when you need an immediate in-memory array.
- `LoadConfig` adds `memmap=True` to document the preferred disk-backed loading path.
- `make_surrogate_data(..., return_generator=True)` is now the default so surrogate lists are not fully materialized in RAM.
- `run_pipeline` drops superseded continuous stages from `PipelineResult` when epoching is enabled (`raw` / `referenced` may be `None`) to reduce peak memory.
- Neuralynx `.ncs` loading rescales to `float32` instead of `float64`.
- `match_elec_names` defaults to `interactive=False` and raises on ambiguous matches (no CI/agent hang).
- `wm_ref` site=`'UI'` now returns a 4-tuple (pads `oob_channels` with `None`) for `ref_mne` unpacking.
- Legacy `make_epochs` supplies default `IED_args` when omitted.
- Parallel defaults for beginner-facing util entry points (`time_resolved_mlm`, connectivity parallel path, burst detection) are `n_jobs=1`; pass `-1` on a cluster.
- `compute_FOOOF_parallel` / `compute_eBOSC_parallel` no longer default to a lab HPC `save_path`.
- Heavy optional imports (`fooof`, `mne_connectivity`, `pycatch22`, IPython) are lazy in utility modules.

### Fixed
- `match_elec_names` and `compute_connectivity` now `raise` exceptions instead of returning exception objects.
- Soft-archived stubs raise `NotImplementedError` with guidance instead of silent `pass` / `None`.

### Planned
- Split mega-modules into `advanced.prep` / `advanced.connectivity` / … and retire `legacy.py` at 2.0.
- Full legacy TFR orchestrator parity behind `TfrConfig` where still needed.
