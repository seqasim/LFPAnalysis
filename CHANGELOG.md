# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [Unreleased]

### Changed
- Default working dtype for signal and TFR arrays is now `float32` (`WORKING_DTYPE`) to roughly halve RAM on local machines.
- `LoadConfig.preload` now defaults to `False` (lazy / disk-backed loads). Set `preload=True` when you need an immediate in-memory array.
- `LoadConfig` adds `memmap=True` to document the preferred disk-backed loading path.
- `make_surrogate_data(..., return_generator=True)` is now the default so surrogate lists are not fully materialized in RAM.
- `run_pipeline` drops superseded continuous stages from `PipelineResult` when epoching is enabled (`raw` / `referenced` may be `None`) to reduce peak memory.
- Neuralynx `.ncs` loading rescales to `float32` instead of `float64`.

### Added
- Modern packaging metadata and optional dependency groups.
- Contributor and governance documentation for open-source collaboration.
- Explicit `joblib` dependency for optional parallel surrogate / stats paths.
