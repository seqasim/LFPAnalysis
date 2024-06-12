# LFPAnalysis

The motivation for this package is that any member of the Saez or Gu lab should be able to:


1. load raw EDF or Neuralynx LFP data into mne (https://mne.tools/stable/index.html)

2. preprocess it according to best practices (to the best of my knowledge and experience)

3. sync it to their behavioral task 

4. extract spectral information with minimal confusion 

5. perform (mostly) non-parametric statistical analyses with this data. 

I am experimenting with making this a fully self-contained Python package that people can download using pip.


The LFPAnalysis folder contains functions split into separate utility libraries depending on what they do. 

The scripts folder contains notebooks of examples using the functions to process our data. 

Eventually, the "data" folder will hold sample test data for people to play around with. 

I have never written a Python package before so I may screw up some logistical aspects of how this is organized or setup. I am generally trying to follow that system laid out here: https://goodresearch.dev/index.html


Most of the Python packages you need here should come from the install for mne: https://mne.tools/stable/install/manual_install.html#manual-install

If missing anything, consult the environment.yml file. 

Some important misc notes if you're on MSSM's MINERVA server (I will compile here as I go and make a more comprehensive guide to using Minerva later): 

1. Make sure your conda environment points away from the default /hpc/user/ directory as this runs out of space quite quickly. See: https://labs.icahn.mssm.edu/minervalab/documentation/conda/

2. If you want to run batch jobs in Jupyter on Minerva: https://labs.icahn.mssm.edu/minervalab/documentation/python-and-jupyter-notebook/

## Installation

```
conda create --name lfp_env pip requests git python=3.10.8
conda activate lfp_env
pip install git+https://github.com/seqasim/LFPAnalysis.git
```

## Updating

```
pip install --upgrade git+https://github.com/seqasim/LFPAnalysis.git
```

## Where to start? 

In the scripts folder you'll find some Jupyter notebooks. Condensed Notebook.ipynb is the best starting point! As I run through different analyses I'll add them into their own notebooks.