# LFPAnalysis

This package was motivated by the need to get data from an .edf or .nlx file into an MNE structure as easily as possible, preprocess it according to best practices, sync it to behavioral data, and extract spectral information fom it. 


The LFPAnalysis folder contains functions split into separate utility libraries depending on what they do. 

The scripts folder contains notebooks of examples using the functions to process our data. 

The "data" folder holds sample test data for people to play around with. 

Most of the Python packages you need here should come from the install for mne: https://mne.tools/stable/install/manual_install.html#manual-install

If missing anything, consult the environment.yml file, which has the fully detailed package information (though not all are necessary).

## Installation

Installation requires git + conda. 
In your terminal, type:

```
cd path_to_install
git clone https://github.com/seqasim/LFPAnalysis.git
conda env create -f environment_manual.yml
```

<!-- ```
conda create --name lfp_env pip requests git python=3.10.8
conda activate lfp_env
pip install git+https://github.com/seqasim/LFPAnalysis.git
``` -->

## Updating

To update the code to reflect changes in the repository:
```
cd path_to_install
git pull
```

<!-- ```
pip install --upgrade --force-reinstall git+https://github.com/seqasim/LFPAnalysis.git 
``` -->

## Where to start? 

In the scripts folder you'll find some Jupyter notebooks. Condensed Notebook.ipynb is the best starting point! As I run through different analyses I'll add them into their own notebooks.