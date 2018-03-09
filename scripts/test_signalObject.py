#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:26:10 2018

@author: hellerc
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/auto/users/hellerc/NEMS/')
from nems.utilities.baphy import baphy_load_dataset, spike_time_to_raster
from nems.signal import Signal
 
# load data in first (create a spike/response matrix)
parmfilepath = '/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m'
options = {'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True}

# load baphy parmfile
out = baphy_load_dataset(parmfilepath,
                           options)
# unpack output
event_times, spike_dict, stim_dict, state_dict = out
event_times['start_index']=[int(x) for x in event_times['start']*100]
event_times['end_index']=[int(x) for x in event_times['end']*100]

# r is response matrix, created from dictionary of spike times
r = spike_time_to_raster(spike_dict=spike_dict,
                         fs=100,
                         event_times=event_times)

# Create the signal object spike raster (r)
sig = Signal(fs=100,
             matrix=r[0],
             name='test_name',
             recording='test_recording',
             epochs=event_times)

# Fold matrix over all stimuli, returning a dictionary where keys are stimuli 
# each value in the dictionary is (reps X cell X bins)
folded_matrix = sig.extract_epochs(stim_dict.keys())

# Average over all reps of each stim and save into dict called psth. Same 
# formate as folded_matrix
psth = dict()
for k in folded_matrix.keys():
    psth[k] = np.mean(folded_matrix[k],
                      axis=0)

# Invert the folding to unwrap the psth back out into a predicted spike_dict by 
# simply replacing all epochs in the signal with their psth
psth_sig = sig.replace_epochs(psth)
