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
import os

from nems.recording import load_recording, get_demo_recordings
import nems.plots.api as nplt
import nems.epoch as ep
from nems import get_setting

font_size=12
params = {'legend.fontsize': font_size-2,
          'figure.figsize': (8, 6),
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

signals_dir = get_setting('NEMS_RECORDINGS_DIR')

# ----------------------------------------------------------------------------
# DATA LOADING & PRE-PROCESSING
#recording.get_demo_recordings(name="TAR010c_272b438ce3a5643e3e474206861096ce3ffdc000.tgz")

uri = os.path.join(signals_dir, "TAR010c_272b438ce3a5643e3e474206861096ce3ffdc000.tgz")

#site = "TAR010c"
# site = "BRT033b"
#site = "bbl099g"

# use this line to load recording from server.
#uri = 'http://hearingbrain.org/tmp/'+site+'.tgz'

# alternatively download the file, save and load from local file:
#filename=site+'.NAT.fs200.tgz'
#recording_path=get_demo_recordings(name=filename)

# uri = '/path/to/recording/' + site + '.tgz'
#uri = os.path.join(recording_path, filename)


rec = load_recording(uri)

epochs = rec.epochs
is_stim = epochs['name'].str.startswith('STIM')
stim_epoch_list = epochs[is_stim]['name'].tolist()
stim_epochs = list(set(stim_epoch_list))
stim_epochs.sort()

epoch = stim_epochs[1]

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

norm_psth = psth - np.mean(psth,axis=1,keepdims=True)
norm_psth /= np.std(psth,axis=1,keepdims=True)
sc = norm_psth @ norm_psth.T / norm_psth.shape[1]


plt.figure()
ax = plt.subplot(2,2,1)
ax.imshow(sc, aspect='equal', cmap='gray')
mm=np.max(sc[sc<1])
i,j=np.where(sc==mm)

ax = plt.subplot(2, 1, 2)
nplt.timeseries_from_vectors(
        psth, fs=resp.fs, ax=ax,
        title="PSTH ({} cells, {} reps)".format(raster.shape[1], raster.shape[0]))


fig=plt.figure()

ax = plt.subplot(3,1,1)
nplt.plot_spectrogram(spec, fs=resp.fs, ax=ax, title=epoch, cmap='gray_r')

ax = plt.subplot(3,1,2)
tt=np.arange(psth.shape[1])/resp.fs-2

ax.plot(tt,psth[25,:], color='black', lw=1)
ax.plot(tt,psth[27,:], color='gray', lw=1)
ax.set_title('sc={:.3f}'.format(sc[25, 27]))
ax.set_ylabel('Spikes/sec')
nplt.ax_remove_box(ax)

ax = plt.subplot(3,1,3)

ax.plot(tt,psth[25, :], color='black', lw=1)
ax.plot(tt,psth[32, :], color='gray', lw=1)
ax.set_title('sc={:.3f}'.format(sc[25, 32]))
ax.set_ylabel('Spikes/sec')
nplt.ax_remove_box(ax)

ax.set_xlabel('Time')

fig.savefig("/Users/svd/Documents/current/ohrc_data_club_2019/example_psth_high_sc.pdf")