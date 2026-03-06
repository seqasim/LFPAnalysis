# Data Model, Supported Files, and Sample Data

## Primary data model

The stable workflow API is built around MNE-Python objects.

- Continuous recordings are represented as `mne.io.Raw`
- Epoched recordings are represented as `mne.Epochs`
- Summary tables are returned as pandas `DataFrame` objects

## Supported input formats

- EDF files
- Neuralynx `.ncs` collections and `.nev` event files
- Existing MNE `.fif` files
- Electrode metadata in CSV or XLSX form

## Electrode metadata contract

### Required columns

- `label`

### Common optional columns

- `x`, `y`, `z`
- `NMM`
- `BN246`
- `YBA_1`
- `collapsed_manual`

## Sample data in this repository

The `data/` directory includes small FIF and table assets that support smoke tests and examples. Use them to validate your environment before pointing the workflow at patient data.

## File-format preparation guide

- Keep one recording session per logical dataset directory.
- Store behavioral timestamps separately from electrophysiology files.
- Normalize electrode labels early so they match recording channel names.
- Keep raw files immutable; save processed outputs as new files.
