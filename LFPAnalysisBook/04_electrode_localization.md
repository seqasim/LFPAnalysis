# Electrode Localization and Metadata Contract

`LFPAnalysis` assumes localization has already happened outside the repository. The package does not localize electrodes from imaging data.

## What the package needs

At minimum, it needs a table with channel labels. Region-of-interest helpers become more useful when you also provide atlas labels and coordinates.

## Recommended workflow

1. Perform localization in your preferred external tool.
2. Export a clean CSV or XLSX file.
3. Ensure the `label` column matches recording channel names or can be matched deterministically.
4. Validate the table with `load_electrode_metadata` before preprocessing.

## Common mistakes

- mixed capitalization across recording and localization labels
- unlabeled microwires embedded in the same table as macro contacts
- multiple atlas columns with unclear precedence
- saving manual relabeling in a notebook instead of the source table
