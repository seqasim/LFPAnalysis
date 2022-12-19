#!/usr/bin/env python
# coding: utf-8

# In[4]:


# load some modules for preprocessing, analyzing and visualizing the data
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import pandas as pd

# install packages to visualize brains and electrode locations
import nilearn
import nimare
import duecredit


# In[5]:


import sys
np.set_printoptions(threshold=sys.maxsize)


# In[4]:


# Get the current working directory
cwd = os. getcwd()
print(cwd)


# In[7]:


# Download additional modules if needed - not necessary to run this every time!
#import sys
#!{sys.executable} -m pip install mne --quiet
#!{sys.executable} -m pip install nilearn --quiet
#!{sys.executable} -m pip install nimare --quiet
#!{sys.executable} -m pip install dipy --quiet


# In[5]:


# Load and visualize coordinates from anatomical reconstruction
anat_file = pd.read_csv(r"MS009labels.csv")
anat_file


# In[6]:


mni_coordinates = anat_file[["mni_x","mni_y","mni_z"]]
mni_coordinates # there are some labeled as unknown in the dataset?

x = mni_coordinates.to_numpy()

from nilearn import plotting
from nimare import utils

plt.figure(figsize=(8, 8))
locs = x
view = plotting.view_markers(locs,
                             marker_labels=['%d'%k for k in np.arange(locs.shape[0])],
                             marker_color='purple',
                             marker_size=5)
view


# In[11]:


# load raw iEEG data (edf format)
raw = mne.io.read_raw_edf('/Users/christinamaher/Desktop/MS009_GemHunters.edf') # our file is in edf format
print(raw) # this will give you a glimpse of the basic details of this raw object (channels x timepoints of sampling; duration; file size)


# In[12]:


# get some info that is saved in the raw.info object (this is similar to a Python dictionary)
n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)
print('The sample data object has {} time samples and {} channels.'''.format(n_time_samps,n_chan))
print(raw.info)
print(ch_names)


# In[13]:


# Use the interactive plot to inspect and mark noisy/irrelevant channels as raw.info['bads'] so they are excluded from later analyses. Note, refer to patient notes to remove epileptic channels 

# this command sets the backend so that the plots are interactive
get_ipython().run_line_magic('matplotlib', 'qt')

# plot all channels so that irrelevant/noisy channels can be saved as bad
raw.plot(n_channels=50)


# In[14]:


# Subset photodiode data and plot - either called DC1 or Research
photodiode_index = ch_names.index("DC1")

photodiode = raw[photodiode_index]

x_photodiode = photodiode[1]
y_photodiode = photodiode[0].T
plt.plot(x_photodiode, y_photodiode)
plt.xlabel("Time")
plt.ylabel("V")
plt.title("Photodiode")


# In[15]:


# load behavior data and save timestamp(s) of interest as variable 
behavior = pd.read_csv(r"/Users/christinamaher/Desktop/MS009.csv")
choice_ts = np.array(behavior['choice_ts'])

# Conver time from ms to s for comparison to photodiode recording 
choice_ts = choice_ts / 1000


# In[16]:


# Overlay timestamps on photodiode data to visualize offset 
zeros = np.array([0] * len(choice_ts))
x_ts = choice_ts
y_ts = zeros.T
plt.scatter(x_ts,y_ts,color='red')


# In[17]:


# This code saves the coordinates of the point you manually press on the plots. More efficient way to compute the offset
photodiode_deflection = plt.ginput(1)

button_press = plt.ginput(1)

# manipulate the inputs that were manually selected so that they can be used to compute offset
photodiode_deflection = photodiode_deflection[0]
photodiode_deflection = photodiode_deflection[0]

button_press = button_press[0]
button_press = button_press[0]


# In[18]:


# Compute offset between ts and photodiode and replot to verify alignment 
offset = button_press - photodiode_deflection
choice_ts = choice_ts - offset
x_ts = choice_ts

plt.plot(x_photodiode, y_photodiode)
plt.scatter(x_ts,y_ts,color='green')


# In[22]:


my_annot = mne.Annotations(onset=choice_ts, duration=0, description=['choice'])
print(my_annot)


# In[23]:


# basic visualizations of EPOCHs objects
raw2 = raw.copy().set_annotations(my_annot)
fig = raw2.plot()


# WIP from here onwards...
# 
# Here you should subset based on a lot of different conditions to create different analysis groups. Dictionary names can be given multiple conditions (win/loss, correct/incorrect, pre-post learning). Also when we load in the behavioral file we can decide on conditions of interest (i.e., training versus not training, learned versus not learned, hint versus no hint) - maybe do this step programmatically so that the interface is nicer?  
# 
# https://mne.tools/0.15/auto_tutorials/plot_artifacts_correction_rejection.html - use this for rejecting bad epochs 
