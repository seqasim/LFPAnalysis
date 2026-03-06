# Contributing to LFPAnalysis

## First-time setup

```bash
git clone https://github.com/seqasim/LFPAnalysis.git
cd LFPAnalysis
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

Conda users can instead run:

```bash
conda env create -f environment-dev.yml
conda activate LFPAnalysis-dev
```

## Development workflow

1. Create a focused branch from the current default branch.
2. Make one logical change per commit.
3. Run the relevant checks before opening a pull request.
4. Update documentation and tests alongside behavior changes.

## Required local checks

```bash
nox -s lint
nox -s tests
```

If your change touches the book or example notebooks, also run:

```bash
nox -s docs
nox -s notebooks
```

## Pull requests

A good pull request includes:

- a concise problem statement
- the user-visible behavior change
- tests for new behavior or regression coverage
- documentation updates where applicable

## Reporting bugs

Please include your operating system, Python version, install method, the input file format you used, and a minimal reproducer.
