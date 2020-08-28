import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
import random

from nems.analysis.gammatone.gtgram import gtgram
from nems import recording, signal
from nems import xforms, get_setting
import nems.gui.editors as gui

# if True, reload all the wav files, generate spectrograms and save as recording
REGENERATE_RECORDING = True

# some hard-coded path specs
BAPHY_ROOT = "/Users/grego/baphy"
RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')
#RECORDING_PATH="/tmp"

recording_file = os.path.join(RECORDING_PATH, "classifier.tgz")

if REGENERATE_RECORDING:
    # generate a couple lists of files
    dir1 = os.path.join(BAPHY_ROOT,"SoundObjects","@FerretVocal","Sounds_set4","*.wav")
    dir2 = os.path.join(BAPHY_ROOT,"SoundObjects","@Speech","sounds","*sa1.wav")
    dir3 = os.path.join(BAPHY_ROOT,"SoundObjects","@Torc","Sounds","*501.wav")

    set1 = glob.glob(dir1)
    set2 = glob.glob(dir2)
    big_set3 = glob.glob(dir3)
    random.shuffle(big_set3)
    torc_set3 = []
    for i in range(len(big_set3)):
        if str.isdigit(big_set3[i][50]) == 1:
            torc_set3.append(big_set3[i])
    random.shuffle(torc_set3)
    set3 = torc_set3[:40]

    # generate a label for each wav file
    sound_classes = np.concatenate((np.zeros(len(set1))+1,
                                    np.zeros(len(set2))+2,
                                    np.zeros(len(set3))+3))
    sound_files = set1 + set2 + set3



    #final sampling rate of spectrogram
    sg_fs = 100

    # arbitrary pre/post stim silence padding
    silence = 0.1

    # iterate through all files, generate spectrogram, stash in dictionary,
    # and generate epochs dictionary along the way.
    sg1 = {}
    epochs = pd.DataFrame(columns=['name','start','end'])
    lastendtime=0.0
    for c, s in zip(sound_classes, sound_files):
        fs, w = wavfile.read(s)
        # add STIM_ to beginning of the base file name so we know what it is
        b = "STIM_" + os.path.basename(s)
        # add pre/post silence
        z = np.zeros(int(fs * silence))

        w = np.concatenate((z,w,z))
        sg1[b] = gtgram(w, fs, 0.02, 1/sg_fs, 18, 200, 20000)

        thisendtime=lastendtime+len(w)/fs
        row = {'name': b, 'start': lastendtime, 'end': thisendtime}
        epochs = epochs.append(row, ignore_index=True)
        row = {'name': f"CLASS_{c:.0f}", 'start': lastendtime+silence, 'end': thisendtime-silence}
        epochs = epochs.append(row, ignore_index=True)

        lastendtime = lastendtime+np.ceil(len(w)/fs * sg_fs)/sg_fs

    # convert stimulus spectrograms into a singal
    stim = signal.TiledSignal(data=sg1, epochs=epochs, fs=sg_fs, name='stim', recording="NAT")
    stim = stim.rasterize()

    # generate a "response" signal for each class
    resp1 = stim.epoch_to_signal("CLASS_1")
    resp2 = stim.epoch_to_signal("CLASS_2")
    resp3 = stim.epoch_to_signal("CLASS_3")

    resp = signal.RasterizedSignal.concatenate_channels([resp1, resp2, resp3])
    resp.name='resp'
    signals = {'stim': stim, 'resp': resp}

    # combine into a recording object
    rec = recording.Recording(signals=signals, name="classifier_data")
    rec.save(recording_file)

else:
    print(f'loading: {recording_file}')
    rec = recording.load_recording(recording_file)

# super simple model architecture
#modelspecname = 'wc.18x2-fir.1x15x2-lvl.2'

# slightly less dumb
modelspecname = 'dlog-wc.18x3.g-fir.1x15x3-lvl.3-dexp.2'

# generate sequence of xforms commands
xfspec = []

meta = {'cellid': 'Classifier', 'batch': 0, 'modelname': modelspecname,
        'recording': 'NAT'}
xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])

xfspec.append(['nems.xforms.mask_for_jackknife',
                {'njacks': 5, 'epoch_regex': '^STIM_'}])
xfspec.append(['nems.xforms.jack_subset', {'keep_only': 1}])

#xfspec.append(['nems.initializers.rand_phi', {'rand_count': 5}])
xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_basic', {'tolerance': 1e-6}])
#xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

xfspec.append(['nems.xforms.predict', {}])
# xfspec.append(['nems.xforms.add_summary_statistics',    {}])
xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspec', 'rec'], ['modelspec']])

# GENERATE PLOTS
xfspec.append(['nems.xforms.plot_summary', {}])

# initialize context
ctx = {'rec': rec}

# actually do the fit
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)

# load in gui
gui.browse_xform_fit(ctx, xfspec)