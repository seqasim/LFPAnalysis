# Installation and Environment Setup

## What this step is for

This chapter gets you into a working environment with the right dependency level for the workflow you want.

## When you should use it

Read this before running any notebook or sample-data workflow.

## Required inputs

- Python 3.10 to 3.12
- the repository clone or the GitHub URL
- enough disk space for optional scientific dependencies

## Minimal example

```bash
git clone https://github.com/seqasim/LFPAnalysis.git
cd LFPAnalysis
pip install -e .[dev]
```

## How to inspect the result

Run `python -c "import LFPAnalysis; print('ok')"` and confirm that the import succeeds.

## Common mistakes

- installing into one environment and running notebooks from another
- using docs-only extras when you need analysis dependencies
- assuming a PyPI release exists already

## Old-to-new translation note

The old repo implicitly relied on a lab-specific environment. The refactored repo separates `analysis`, `docs`, `test`, and `dev` extras so you can install only what you need.

## Quick install paths

- everyday contributor path: `pip install -e .[dev]`
- documentation only: `pip install -e .[docs]`
- analysis workflows: `pip install -e .[analysis]`

Next step: {doc}`02_data_model`
