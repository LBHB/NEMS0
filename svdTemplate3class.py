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
import nems.gui.editors as gui
import nems
import random

# if True, reload all the wav files, generate spectrograms and save as recording
REGENERATE_RECORDING = True

# some hard-coded path specs
BAPHY_ROOT = "/Users/grego/baphy"
RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')
#RECORDING_PATH="/tmp"

recording_file = os.path.join(RECORDING_PATH, "classifier.tgz")
class_labels = ['Ferret', 'Speech','Torc']

if not os.path.exists(recording_file):
    REGENERATE_RECORDING = True

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
    max_sg_freq = 8000
    for c, s in zip(sound_classes, sound_files):
        fs, w = wavfile.read(s)

        if fs < max_sg_freq*2:  # ie, less than half of max spectrogram frequency
            desired_bins = int(len(w) / fs * max_sg_freq*2)
            w = resample(w, desired_bins)
            fs = max_sg_freq*2
            print(f'resampled {s} to {desired_bins} bins')

        b = "STIM_" + os.path.basename(s)
        # add pre/post silence
        z = np.zeros(int(fs * silence))

        w = np.concatenate((z, w, z))
        sg1[b] = gtgram(w, fs, 0.02, 1/sg_fs, 18, 100, 8000)

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

n_classes = 3
# single filter classifier. pretty dumb
modelspecname = f'dlog-wc.18x2.g-fir.2x15-wc.1x{n_classes}.z-lvl.{n_classes}-dexp.{n_classes}'

# slightly less dumb. Recommend starting here
modelspecname = f'dlog-wc.18x{n_classes}.g-fir.1x15x{n_classes}-lvl.{n_classes}-dexp.{n_classes}'

# consider trying this with more classes
n_filters = 3
modelspecname = f'dlog-wc.18x{n_filters}.g-fir.1x15x{n_filters}-wc.{n_filters}x{n_classes}.z-lvl.{n_classes}-dexp.{n_classes}'

modelspecname = 'dlog-wc.18x3.g-fir.1x15x3-lvl.3-dexp.2'

# generate sequence of xforms commands
xfspec = []

meta = {'cellid': 'Classifier', 'batch': 0, 'modelname': modelspecname,
        'recording': 'NAT'}

# initialize context
ctx = {'rec': rec}

# xforms logic:
# define ctx dictionary.
# each xforms command takes in ctx and returns updated ctx
#ctx.update(xforms.init_from_keywords(keywordstring=modelspecname, meta=meta, **ctx))
from nems.initializers import from_keywords
ctx['modelspec'] = from_keywords(keyword_string=modelspecname,
                                       meta=meta, registry=None, rec=None,
                                       input_name='stim',
                                       output_name='resp')

# separate out est and val
ctx.update(xforms.mask_for_jackknife(njacks=5, epoch_regex='^STIM_', **ctx))

# quick and dirty throw away all jackknife sets but one to produce a single
# est/val pair of recordings
ctx.update(xforms.jack_subset(keep_only=1, **ctx))

#xfspec.append(['nems.initializers.rand_phi', {'rand_count': 5}])
ctx.update(xforms.fit_basic_init(**ctx))

ctx.update(xforms.fit_basic(tolerance=1e-6, **ctx))
#xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

ctx.update(xforms.predict(**ctx))
ctx.update(xforms.add_summary_statistics(**ctx))

# non-xforms command, non-standard way of calling it
#ctx['modelspec'] = nems.analysis.api.standard_correlation(
#    ctx['est'], ctx['val'], ctx['modelspec'], ctx['rec'])

# GENERATE PLOTS
ctx.update(xforms.plot_summary(**ctx))

# load in gui
gui.browse_xform_fit(ctx, xfspec)