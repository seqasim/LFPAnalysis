# Testing Guide

## Local commands

```bash
nox -s lint
nox -s tests
nox -s docs
nox -s notebooks
```

## Pytest markers

- `unit`: fast tests for validation, schemas, and pure transforms
- `integration`: multi-step workflow tests, often involving synthetic MNE objects
- `notebook`: notebook execution coverage
- `slow`: heavier tests you may want to exclude during quick iteration
- `optional_dep`: tests that require optional analysis extras

## Recommended quick loop

```bash
pytest -m "unit and not optional_dep"
```

## CI-equivalent test command

```bash
pytest -m "not notebook and not slow" --cov=LFPAnalysis.workflow --cov-fail-under=80
```

## Fixture rules

- Keep canonical fixture files under `tests/data/`
- Prefer synthetic MNE objects built in fixtures over large binary assets
- Use bundled repository sample data only for docs or notebook smoke tests

## Notebook runtime expectations

Smoke notebooks in `LFPAnalysisBook/smoke-tests/` should stay deterministic and bounded. They are expected to validate imports, sample-data loading, and one small example per workflow area rather than act as full tutorials.
