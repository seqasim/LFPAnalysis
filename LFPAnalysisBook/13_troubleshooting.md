# Troubleshooting, FAQ, and Glossary

## FAQ

### Why does a workflow step ask for optional dependencies?

Some analysis steps, especially connectivity and FOOOF, live behind the `analysis` extra to keep the base install lighter.

### Why is `laplacian` in the API if it is not available yet?

The workflow registry includes reserved entries where the legacy implementation is incomplete. This makes the supported surface explicit instead of silently pretending the option exists.

### Why do my channels not reference correctly?

Check the electrode metadata first. Most reference failures come from mismatched channel labels or missing localization rows.

## Glossary

- `Raw`: continuous MNE data object
- `Epochs`: event-locked MNE data object
- `reference`: the signal subtracted from each electrode channel
- `baseline`: the comparison period used to normalize a signal or feature
- `artifact`: non-neural contamination or unusable signal segment
- `IED`: interictal epileptiform discharge
