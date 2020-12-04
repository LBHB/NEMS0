import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample

from nems.analysis.gammatone.gtgram import gtgram
from nems import recording, signal
from nems import xforms, get_setting
from nems import get_setting
import nems.gui.editors as gui
import nems
import random
import nems.tf.cnnlink_new as cnn

# RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')
RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')
recording_file = os.path.join(RECORDING_PATH, "classifier.tgz")

use_new_stim = True
if use_new_stim:
    soundpath = "/Users/grego/OneDrive/Documents/Sounds/Classifier"

    class_labels = ['Animal Sounds', 'Environment', 'Machine', 'Marmoset Vocalizations',
                    'Music', 'Speech', 'Transients']
    cats = len(class_labels)

    sound_sets = {}
    for cc in class_labels:
        if len(glob.glob(f'{soundpath}/{cc}/*.wav')) < 25:
            sound_sets[cc] = glob.glob(f"{soundpath}/{cc}/*.wav")
        else:
            sound_sets[cc] = glob.glob(f"{soundpath}/{cc}/*.wav")[:25]

    lens = [len(sound_sets[i]) for i in sound_sets.keys()]
    sound_classes = np.repeat(np.arange(1, len(lens) + 1), lens)
    sound_files = []
    for zz in sound_sets.values():
        sound_files = sound_files + zz

else:
    ##Old classifier than works
    BAPHY_ROOT = "/Users/grego/baphy"
    RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')

    recording_file = os.path.join(RECORDING_PATH, "classifier.tgz")
    class_labels = ['Ferret', 'Speech', 'Torc']

    # generate a couple lists of files
    dir1 = os.path.join(BAPHY_ROOT, "SoundObjects", "@FerretVocal", "Sounds_set4", "*.wav")
    dir2 = os.path.join(BAPHY_ROOT, "SoundObjects", "@Speech", "sounds", "*sa1.wav")
    dir3 = os.path.join(BAPHY_ROOT, "SoundObjects", "@Torc", "Sounds", "*501.wav")

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
    sound_classes = np.concatenate((np.zeros(len(set1)) + 1,
                                    np.zeros(len(set2)) + 2,
                                    np.zeros(len(set3)) + 3))
    sound_files = set1 + set2 + set3

# final sampling rate of spectrogram
sg_fs = 100

# arbitrary pre/post stim silence padding
silence = 0.1

# iterate through all files, generate spectrogram, stash in dictionary,
# and generate epochs dictionary along the way.
sg1 = {}
epochs = pd.DataFrame(columns=['name', 'start', 'end'])
lastendtime = 0.0
max_sg_freq = 8000
min_stim_len = 4.0
min_stim_samples = min_stim_len * max_sg_freq
for c, s in zip(sound_classes, sound_files):
    fs, w = wavfile.read(s)

    if fs < max_sg_freq * 2:  # ie, less than half of max spectrogram frequency
        desired_bins = int(len(w) / fs * max_sg_freq * 2)
        w = resample(w, desired_bins)
        fs = max_sg_freq * 2
        print(f'resampled {s} to {desired_bins} bins')

    b = "STIM_" + os.path.basename(s)

    # pad if necessary
    desired_len = int(min_stim_len * fs)
    if len(w) < desired_len:
        print(f"padding {s} with {desired_len - len(w)} samples")
        p = np.zeros(desired_len - len(w))
    elif len(w) > desired_len:
        print(f"trimming {s} by {len(w) - desired_len} samples")
        w = w[:desired_len]
        p = np.array([])
    else:
        p = np.array([])
    # add pre/post silence
    z = np.zeros(int(fs * silence))

    w = np.concatenate((z, w, p, z))
    sg1[b] = gtgram(w, fs, 0.02, 1 / sg_fs, 18, 100, 8000)

    thisendtime = lastendtime + len(w) / fs
    row = {'name': b, 'start': lastendtime, 'end': thisendtime}
    epochs = epochs.append(row, ignore_index=True)
    row = {'name': f"CLASS_{c:.0f}", 'start': lastendtime + silence, 'end': thisendtime - silence}
    epochs = epochs.append(row, ignore_index=True)
    row = {'name': "REFERENCE", 'start': lastendtime, 'end': thisendtime}
    epochs = epochs.append(row, ignore_index=True)

    lastendtime = lastendtime + np.ceil(len(w) / fs * sg_fs) / sg_fs

# convert stimulus spectrograms into a singal
stim = signal.TiledSignal(data=sg1, epochs=epochs, fs=sg_fs, name='stim', recording="NAT")
stim = stim.rasterize()

# generate a "response" signal for each class
if use_new_stim == True:
    resp1 = stim.epoch_to_signal("CLASS_1")
    resp2 = stim.epoch_to_signal("CLASS_2")
    resp3 = stim.epoch_to_signal("CLASS_3")
    resp4 = stim.epoch_to_signal("CLASS_4")
    resp5 = stim.epoch_to_signal("CLASS_5")
    resp6 = stim.epoch_to_signal("CLASS_6")
    resp7 = stim.epoch_to_signal("CLASS_7")
    resp = signal.RasterizedSignal.concatenate_channels([resp1, resp2, resp3, resp4, resp5, resp6, resp7])
    n_classes, n_filters = 7, 7

else:
    resp1 = stim.epoch_to_signal("CLASS_1")
    resp2 = stim.epoch_to_signal("CLASS_2")
    resp3 = stim.epoch_to_signal("CLASS_3")
    resp = signal.RasterizedSignal.concatenate_channels([resp1, resp2, resp3])
    n_classes, n_filters = 3, 3

resp.name = 'resp'
signals = {'stim': stim, 'resp': resp}

# combine into a recording object
rec = recording.Recording(signals=signals, name="classifier_data")
rec.save(recording_file)

modelspecname = f'dlog-wc.18x{n_classes}.g-fir.1x15x{n_classes}-lvl.{n_classes}-dexp.{n_classes}'

# initialize context
ctx = {'rec': rec}
meta = {'cellid': 'Classifier', 'batch': 0, 'modelname': modelspecname,
        'recording': 'NAT'}

ctx.update(xforms.init_from_keywords(keywordstring=modelspecname, meta=meta, **ctx))

# separate out est and val
ctx.update(xforms.mask_for_jackknife(njacks=5, epoch_regex='^STIM_', **ctx))

# make single est/val set, throw away other jackknifes
ctx.update(xforms.jack_subset(keep_only=1, **ctx))

# xfspec.append(['nems.initializers.rand_phi', {'rand_count': 5}])
# ctx.update(xforms.fit_basic_init(**ctx))
# ctx.update(xforms.fit_basic(tolerance=1e-6, **ctx))
ctx.update(cnn.fit_tf_init(max_iter=1000, early_stopping_tolerance=5e-4, use_modelspec_init=True, **ctx))
ctx.update(cnn.fit_tf(max_iter=1000, early_stopping_tolerance=1e-4, use_modelspec_init=True, **ctx))

ctx.update(xforms.predict(**ctx))
ctx.update(xforms.add_summary_statistics(**ctx))

# Make xfspec anyway but don't run it, GUI seems to like it
xfspec = []
xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])
xfspec.append(['nems.tf.cnnlink_new.fit_tf_init',
               {'max_iter': 1000, 'early_stopping_tolerance': 5e-4, 'use_modelspec_init': True}])
xfspec.append(['nems.tf.cnnlink_new.fit_tf',
               {'max_iter': 1000, 'early_stopping_tolerance': 1e-4, 'use_modelspec_init': True}])
xfspec.append(['nems.xforms.predict', {}])
xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspec', 'rec'], ['modelspec']])
xfspec.append(['nems.xforms.plot_summary', {}])

# GENERATE PLOTS
ctx.update(xforms.plot_summary(**ctx))

# load in gui
gui.browse_xform_fit(ctx, xfspec)