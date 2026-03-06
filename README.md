# LFPAnalysis

LFPAnalysis is a Python toolkit for turning human intracranial and local field potential recordings into MNE-native objects, preprocessing them reproducibly, synchronizing them with behavioral data, and running common downstream analyses such as referencing, baselining, PSD/FOOOF, time-frequency analysis, connectivity, and statistics.

## Why this repository exists

LFPAnalysis is aimed at researchers who need a practical bridge from raw recording formats to analysis-ready MNE objects without rebuilding the same preprocessing stack for each project. The repository now separates a stable beginner-facing workflow from the lower-level utility modules used to support more custom pipelines.

## Features

- Load EDF, Neuralynx, and MNE-native data into a consistent workflow.
- Preprocess, reference, baseline, epoch, and synchronize signals with behavioral events.
- Run spectral, oscillatory, connectivity, and statistics utilities built for iEEG/LFP workflows.
- Learn the package through a structured Jupyter Book and smoke-test notebooks.
- Reuse bundled sample data for quick validation and onboarding.

## Quick install

### Standard pip install

A PyPI release is planned. Until then, install directly from GitHub:

```bash
pip install git+https://github.com/seqasim/LFPAnalysis.git
```

### Contributor install

```bash
git clone https://github.com/seqasim/LFPAnalysis.git
cd LFPAnalysis
pip install -e .[dev]
```

### Conda install

```bash
git clone https://github.com/seqasim/LFPAnalysis.git
cd LFPAnalysis
conda env create -f environment.yml
conda activate LFPAnalysis
pip install -e .
```

## Five-minute quickstart

```python
from pathlib import Path

from LFPAnalysis import (
    ArtifactConfig,
    BaselineConfig,
    EpochConfig,
    LoadConfig,
    PipelineConfig,
    ReferenceConfig,
    SpectralConfig,
    run_pipeline,
)

root = Path("data")
config = PipelineConfig(
    load=LoadConfig(path=root / "sample_ieeg.fif", file_format="mne"),
    reference=ReferenceConfig(method="none"),
    artifact=ArtifactConfig(methods=["none"]),
    baseline=BaselineConfig(mode="zscore", enabled=False),
    epoch=EpochConfig(enabled=False),
    spectral=SpectralConfig(enabled=False),
)
result = run_pipeline(config)
print(result.raw.info["sfreq"])
```

## Supported inputs and outputs

### Inputs

- EDF recordings
- Neuralynx `.ncs` and `.nev` recordings
- MNE `Raw` or `Epochs` objects
- Electrode metadata in CSV or XLSX format

### Outputs

- MNE `Raw` and `Epochs` objects
- pandas tables for artifacts and baselining summaries
- spectral and connectivity summary objects from analysis workflows

## Documentation

The canonical onboarding path is the `LFPAnalysisBook/` directory. It covers installation, data contracts, quickstarts, preprocessing choices, artifact handling, baselining, connectivity, statistics, and troubleshooting.

## Contributing and citation

- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Citation metadata: `CITATION.cff`
- Changelog: `CHANGELOG.md`
