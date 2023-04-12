# LFPAnalysis

The general motivation for this package is that any member of the Saez or Gu lab should be able to load their raw EDF or NLX LFP data into mne, 
preprocess it, sync it to their behavioral task, and perform spectral analyses with minimal confusion. 

To that end, I am experimenting with making this a fully self-contained Python package that people can download using pip.

The LFPAnalysis folder contains functions split into separate utility libraries depending on what they do. 

The scripts folder contains notebooks of examples using the functions to process our data. 

Eventually, I will add a "data" folder with test data for people to play around with. 

I have never written a Python package before so I may screw up some logistical aspects of how this is organized or setup. I am generally trying to follow that system laid out here: https://goodresearch.dev/index.html

Things that this package should allow one to do: 

1. Load raw data (ephys + electrode info)
2. Preprocess (filter noise, remove bad data, re-reference)
3. Ripple detection 
4. Oscillatory burst detection 

Most of the Python packages you need here should come from the install for mne: https://mne.tools/stable/install/manual_install.html#manual-install

Otherwise, consult the environment.yml file. 

Some important misc notes if you're on MINERVA (I will compile here as I go and make a more comprehensive guide to using Minerva later): 

1. Make sure your conda environment points away from the default /hpc/user/ directory as this runs out of space quite quickly. See: https://labs.icahn.mssm.edu/minervalab/documentation/conda/

2. If you want to run batch jobs in Jupyter on Minerva: https://labs.icahn.mssm.edu/minervalab/documentation/python-and-jupyter-notebook/

## Installation

```
conda create --name lfp_env pip requests python=3.10.8
conda activate lfp_env
pip install git+https://github.com/seqasim/LFPAnalysis.git
```