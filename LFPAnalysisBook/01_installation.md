# Installation and Environment Setup

## Supported Python versions

`LFPAnalysis` currently supports Python 3.10 through 3.12.

## Quick install paths

### Source install from GitHub

```bash
git clone https://github.com/seqasim/LFPAnalysis.git
cd LFPAnalysis
pip install -e .
```

### Contributor install

```bash
pip install -e .[dev]
pre-commit install
```

### Conda install

```bash
conda env create -f environment.yml
conda activate LFPAnalysis
pip install -e .
```

## Optional dependency groups

- `analysis`: FOOOF, connectivity, tensorpac, and related scientific extras
- `docs`: Jupyter Book and documentation tooling
- `test`: pytest, coverage, and notebook smoke-test tooling
- `dev`: everything above plus formatting and pre-commit hooks

## Installation troubleshooting

### `ModuleNotFoundError: mne`

Install the package into your active environment with `pip install -e .` or `pip install -e .[dev]`.

### Notebook build failures

Install docs extras with `pip install -e .[docs]`.

### Connectivity or FOOOF import failures

Install analysis extras with `pip install -e .[analysis]`.
