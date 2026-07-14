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
conda env create -f environment.yml      # this already runs `pip install -e .`
conda activate LFPAnalysis
python -m ipykernel install --user --name LFPAnalysis --display-name "LFPAnalysis"
```

In Jupyter or VS Code, select the **LFPAnalysis** kernel before running any notebook.

## How to inspect the result

Verify the install from a neutral directory (not the repo root), so a broken or cwd-masked install fails loudly:

```bash
cd ~
python -c "import sys, LFPAnalysis; print(sys.executable); print(LFPAnalysis.__file__)"
```

Confirm that `sys.executable` points into the `LFPAnalysis` conda env and that `LFPAnalysis.__file__` resolves under your clone.

## Common mistakes

- installing into one environment and running notebooks from another — select the `LFPAnalysis` kernel
- trusting an `import LFPAnalysis` that only works from inside the repo directory (cwd is on `sys.path` and can hide a broken editable install)
- installing extras you do not need, or skipping analysis extras when you need FOOOF / connectivity
- assuming a PyPI release exists already

## Old-to-new translation note

The old repo implicitly relied on a lab-specific environment. The refactored repo separates `analysis`, `docs`, `test`, and `dev` extras so you can install only what you need.

## Quick install paths

- recommended conda path: `conda env create -f environment.yml` (already includes the editable install)
- everyday pip contributor path: `pip install -e .[dev]`
- documentation only: `pip install -e .[docs]`
- analysis workflows: `pip install -e .[analysis]`

If imports fail after install, see {doc}`30_troubleshooting`.

Next step: {doc}`02_data_model`
