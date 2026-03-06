# Artifact and IED Detection

The workflow layer standardizes artifact outputs as long-form pandas tables.

## Built-in detector registry

- `none`
- `misc`
- `ied`
- `custom`

## Choosing an artifact policy

### `misc`

Use for broad gradient-based artifact detection when you want a conservative first-pass quality screen.

### `ied`

Use when your task explicitly requires detecting interictal epileptiform discharges.

### `custom`

Use when you need a lab-specific detector but still want standardized output tables.

## Output schema

Artifact tables use these columns:

- `event_kind`
- `channel`
- `time_seconds`
- `sample_index`

## Artifact decision guide

- start with `misc` on new datasets
- add `ied` only if the scientific question or clinical context needs it
- save detector thresholds with the analysis result metadata
