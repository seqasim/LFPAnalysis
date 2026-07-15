# Notebook Conventions

Apply to every notebook under `notebooks/<slug>/`.

## Cell order

1. **Markdown — Goal** (1–3 sentences: what this notebook computes)
2. **Code — Paths** (hybrid PATHS block below)
3. **Markdown — Assumptions** (stage, event, bands, regressors from Dataset Profile)
4. **Code — Analysis** (adapted from canonical book / worked-example sources)
5. **Code — Quick plot or print inspection** (optional but preferred)
6. **Markdown — Next steps** (link to next notebook in the bundle + book chapter paths)

## Hybrid PATHS block (required)

Use this pattern first. Adapt keys to the Dataset Profile; omit unused modalities. All later cells must load via `P[...]`, never hard-coded absolute user paths.

```python
from pathlib import Path

# --- USER PATHS (fill these) ---
USE_SAMPLE_DATA = True  # set False after filling PATHS
PATHS = {
    "epochs": Path(""),          # e.g. .../sub-01_feedback-epo.fif
    "behavior": Path(""),        # e.g. .../sub-01_beh.csv
    "electrodes": Path(""),      # optional ROI/labels table
}
SAMPLE_PATHS = {
    "epochs": Path("../../data/sample_feedback_start-epo.fif"),
    "behavior": Path("../../data/sample_beh.csv"),
    "electrodes": Path("../../data/sample_labels_bp"),
}
P = SAMPLE_PATHS if USE_SAMPLE_DATA else PATHS
```

### Path depth

Notebooks live at `notebooks/<slug>/…ipynb`, so sample data is `../../data/...` relative to the notebook file.

If scaffolding continuous / non-feedback analyses, point `SAMPLE_PATHS` at the closest sample files under `data/` (e.g. `sample_ieeg.fif`, `sample_beh.csv`) and note the mismatch in the Assumptions cell.

### Validation cell (recommended after PATHS)

```python
for key, path in P.items():
    if path is None or str(path) == ".":
        continue
    if not Path(path).expanduser().exists():
        raise FileNotFoundError(
            f"Missing {key}: {path}. Fill PATHS or keep USE_SAMPLE_DATA=True."
        )
print("Using", "SAMPLE_PATHS" if USE_SAMPLE_DATA else "PATHS")
for k, v in P.items():
    print(f"  {k}: {v}")
```

## Loading patterns

**Epochs (preferred):**

```python
import pandas as pd
from LFPAnalysis import load_lfp
from LFPAnalysis.config import LoadConfig

beh = pd.read_csv(P["behavior"])
epochs = load_lfp(LoadConfig(path=P["epochs"], file_format="mne", preload=True))
# Attach only columns that exist / are needed:
epochs.metadata = beh[["reward", "rpe"]].copy()  # adapt to Dataset Profile
```

Use `preload=True` whenever you will `.pick()`, filter, or run spectral/connectivity helpers. If `load_lfp` is awkward for a given file, `mne.read_epochs(P["epochs"], preload=True)` is acceptable (as in book chapter 10b).

**Electrodes:**

```python
elec_df = pd.read_csv(P["electrodes"])
# expect a label column and ROI column(s); adapt names from Dataset Profile
```

## Stats dataframe columns

When building tables for `statistics_utils`, include:

| Column | Role |
|--------|------|
| `participant` | Subject id |
| `unique_label` | Electrode / bipolar id (unique across subjects if concatenated) |
| `trial` | Trial index within participant |
| `ts` | Time (seconds) for time-resolved models |
| `tfr` or `power` | Dependent measure |
| regressors | e.g. `rpe`, `reward` |

## Style

- Keep notebooks focused: one recipe's job per file
- Prefer small channel/frequency subsets in dry-run comments for speed
- Seed numpy before permutation stats: `np.random.seed(42)`
- Use `verbose=False` on noisy MNE calls in scaffold cells
- Do not commit execution outputs; leave `execution_count` null

## Optional profile file

If helpful, write `notebooks/<slug>/DATASET_PROFILE.md` with the intake summary so later sessions reuse the same assumptions.
