# Quickstart: Raw File to First Result

The stable public entry point is `run_pipeline`, backed by typed config objects.

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

root = Path("../data")
config = PipelineConfig(
    load=LoadConfig(path=root / "sample_ieeg.fif", file_format="mne"),
    reference=ReferenceConfig(method="none"),
    artifact=ArtifactConfig(methods=["none"]),
    baseline=BaselineConfig(mode="none", enabled=False),
    epoch=EpochConfig(enabled=False),
    spectral=SpectralConfig(enabled=False),
)
result = run_pipeline(config)
```

## What this gives you

- `result.raw`: the loaded MNE object
- `result.referenced`: the post-reference data object
- `result.artifact_tables`: standardized event tables keyed by detector name
- `result.baseline_summary`: per-channel baseline diagnostics
- `result.spectral`: optional spectral outputs

## When to drop into advanced utilities

Use the workflow API first. Drop to `LFPAnalysis.lfp_preprocess_utils`, `analysis_utils`, or `oscillation_utils` only when you need a lab-specific or analysis-specific customization that the stable API does not expose yet.
