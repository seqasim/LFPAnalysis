# Time-Frequency Analysis

Time-frequency transforms are useful once event alignment and baseline choices are already defensible.

## Recommended order of operations

1. Validate loading and channel metadata
2. Choose reference and artifact policies
3. Confirm epoch timing
4. Choose baseline mode
5. Run TFR analysis

## Practical note

The repository still exposes lower-level time-frequency utilities in `lfp_preprocess_utils`. Use the smoke notebook in this chapter to confirm that your environment can compute a bounded test transform before scaling up.
