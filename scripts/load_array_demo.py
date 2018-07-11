#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for loading NEMS multichannel recording.

Requires NEMS toolbox (and dependencie) to load. Pull the master branch from
here:
    https://github.com/LBHB/NEMS/

Created on Fri May 11 14:52:14 2018

@author: svd
"""

import matplotlib.pyplot as plt
import numpy as np

from nems.recording import load_recording
import nems.plots.api as nplt
import nems.epoch as ep

# site = "TAR010c"
# site = "BRT033b"
site = "bbl099g"

# use this line to load recording from server.
uri = 'http://hearingbrain.org/tmp/'+site+'.tgz'

# alternatively download the file, save and load from local file:
# uri = '/path/to/recording/' + site + '.tgz'

rec = load_recording(uri)

epochs = rec.epochs
is_stim = epochs['name'].str.startswith('STIM')
stim_epoch_list = epochs[is_stim]['name'].tolist()
stim_epochs = list(set(stim_epoch_list))
stim_epochs.sort()

epoch = stim_epochs[11]

resp = rec['resp'].rasterize()
stim = rec['stim']

# get big stimulus and response matrices. Note that "time" here is real
# experiment time, so repeats of the test stimulus show up as single-trial
# stimuli.
X = stim.rasterize().as_continuous()
Y = resp.as_continuous()

print('Single-trial stim/resp data extracted to X/Y matrices!')

# here's how you get rasters aligned to each stimulus:
# define regex for stimulus epochs
epoch_regex = '^STIM_'
epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)
folded_resp = resp.extract_epochs(epochs_to_extract)

print('Response to each stimulus extracted to folded_resp dictionary')


# or just plot the PSTH for an example stimulus
raster = resp.extract_epoch(epoch)
psth = np.mean(raster, axis=0)
spec = stim._data[epoch]

plt.figure()
ax = plt.subplot(2, 1, 1)
nplt.plot_spectrogram(spec, fs=resp.fs, ax=ax, title=epoch)

ax = plt.subplot(2, 1, 2)
nplt.timeseries_from_vectors(
        psth, fs=resp.fs, ax=ax,
        title="PSTH ({} cells, {} reps)".format(raster.shape[1], raster.shape[0]))

plt.tight_layout()
