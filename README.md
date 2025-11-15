# LFPAnalysis

This package was motivated by the need to get data from an .edf or .nlx file into an MNE structure as easily as possible, preprocess it according to best practices, sync it to behavioral data, and extract spectral information fom it. 


The LFPAnalysis folder contains functions split into separate utility libraries depending on what they do. 

The scripts folder contains notebooks of examples using the functions to process our data. 

The "data" folder holds sample test data for people to play around with. 

Most of the Python packages you need here should come from the install for mne: https://mne.tools/stable/install/manual_install.html#manual-install

If missing anything, consult the environment.yml file, which has the fully detailed package information (though not all are necessary).

## Installation

### Option 1: Using pip (Recommended)

The easiest way to install LFPAnalysis is using pip:

```bash
pip install git+https://github.com/seqasim/LFPAnalysis.git
```

Or if you want to install in development mode (editable):

```bash
git clone https://github.com/seqasim/LFPAnalysis.git
cd LFPAnalysis
pip install -e .
```

### Option 2: Using conda

If you prefer using conda:

```bash
cd path_to_install
git clone https://github.com/seqasim/LFPAnalysis.git
conda env create -f environment_manual.yml
conda activate LFPAnalysis
pip install -e .
```

**Note:** You'll need to install the package itself after creating the conda environment using `pip install -e .` from the repository directory.

## Updating

### If installed via pip:

```bash
pip install --upgrade --force-reinstall git+https://github.com/seqasim/LFPAnalysis.git
```

### If installed from source (git clone):

```bash
cd path_to_install/LFPAnalysis
git pull
pip install -e .  # Reinstall to pick up changes
```

## Testing

To run the test suite, first make sure you have the package installed and pytest available:

```bash
pip install pytest
pytest tests/
```

Or run with more verbose output:

```bash
pytest tests/ -v
```

## Where to start? 

In the scripts folder you'll find some Jupyter notebooks. Condensed Notebook.ipynb is the best starting point! As I run through different analyses I'll add them into their own notebooks.