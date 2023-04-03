# TODO: tests for utility functions in nems/plots/*, like those in nems/plots/assemble.py

import os
import numpy as np
import matplotlib.pyplot as plt

import nems0.recording as recording
import nems0.plots.api as nplt
import nems0.epoch as ep
import nems0

signals_dir = nems0.get_setting('NEMS_RECORDINGS_DIR')

#uri = signals_dir + "/por074b-c2.tgz"
#uri = signals_dir + "/BRT026c-02-1.tgz"
#cellid = "BRT026c-02-1"
recording_file = "TAR010c.NAT.fs100.ch18.tgz"
uri = os.path.join(signals_dir, recording_file)

cellid = "TAR010c-18-2"

def test_plots():
    recording.get_demo_recordings(name=recording_file)
    rec = recording.load_recording(uri)

    resp = rec['resp'].rasterize()
    stim = rec['stim'].rasterize()

    epoch_regex = "^STIM_"

    stim_epochs = ep.epoch_names_matching(resp.epochs, epoch_regex)

    r = resp.as_matrix(stim_epochs) * resp.fs
    s = stim.as_matrix(stim_epochs)
    repcount = np.sum(np.isfinite(r[:, :, 0, 0]), axis=1)
    max_rep_id, = np.where(repcount == np.max(repcount))

    t = np.arange(r.shape[-1]) / resp.fs

    plt.figure()

    ax = plt.subplot(3, 1, 1)
    nplt.plot_spectrogram(s[max_rep_id[-1],0,:,:], fs=stim.fs, ax=ax,
                          title="cell {} - stim".format(cellid))

    ax = plt.subplot(3, 1, 2)
    nplt.raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster')

    ax = plt.subplot(3, 1, 3);
    nplt.psth_from_raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster',
                          ylabel='spk/s')

    plt.tight_layout()
