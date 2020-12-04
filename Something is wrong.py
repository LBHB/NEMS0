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

#New loading of my files that doesn't
RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')
recording_file = os.path.join(RECORDING_PATH, "classifier.tgz")
class_labels = ['Animal Sounds','Music','Speech']
cats = len(class_labels)

sound_sets = {}
for cc in class_labels:
    sound_sets[cc] = glob.glob(f"/Users/grego/OneDrive/Documents/Sounds/Classifier/{cc}/*.wav")[:40]

lens = [len(sound_sets[i]) for i in sound_sets.keys()]
sound_classes = np.repeat(np.arange(1, len(lens)+1), lens)
sound_files = []
for zz in sound_sets.values():
    sound_files = sound_files + zz


##Old classifier than works
BAPHY_ROOT = "/Users/grego/baphy"
RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')

recording_file = os.path.join(RECORDING_PATH, "classifier.tgz")
class_labels = ['Ferret', 'Speech','Torc']

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

#####################################################
#This part is the same regardless of which you loaded,
#but only the latter one works for the model.

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
    row = {'name': f"CLASS_{c:.0f}", 'start': lastendtime + silence, 'end': thisendtime - silence}
    epochs = epochs.append(row, ignore_index=True)
    row = {'name': "REFERENCE", 'start': lastendtime + silence, 'end': thisendtime - silence}
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


# super simple model architecture
#modelspecname = 'wc.18x2-fir.1x15x2-lvl.2'

n_classes = 3
n_filters = 3

# single filter classifier. pretty dumb
# modelspecname = f'dlog-wc.18x2.g-fir.2x15-wc.1x{n_classes}.z-lvl.{n_classes}-dexp.{n_classes}'

# slightly less dumb. Recommend starting here
# modelspecname = f'dlog-wc.18x{n_classes}.g-fir.1x15x{n_classes}-lvl.{n_classes}-dexp.{n_classes}'

# consider trying this with more classes
# modelspecname = f'dlog-wc.18x{n_filters}.g-fir.1x15x{n_filters}-wc.{n_filters}x{n_classes}.z-lvl.{n_classes}-dexp.{n_classes}'

modelspecname = f'dlog-wc.18x{n_classes}.g-fir.1x15x{n_classes}-lvl.{n_classes}-dexp.{n_classes}'
#######
#########
# generate sequence of xforms commands
# xfspec = []
#
# meta = {'cellid': 'Classifier', 'batch': 0, 'modelname': modelspecname,
#         'recording': 'NAT'}
# # generate modelspec
# xfspec = []
# load_command = 'nems.demo.loaders.demo_loader'
#
# # load internally:
# #xfspec.append(['nems.xforms.load_recordings',
# #               {'recording_uri_list': [recording_uri]}])
# # load from external format
# xfspec.append(['nems.xforms.load_recording_wrapper',
#                {'load_command': load_command,
#                 'exptid': 'classifier',
#                 'datafile': recording_file}])
# xfspec.append(['nems.xforms.split_by_occurrence_counts',
#                {'epoch_regex': '^STIM_'}])
# xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])
# ##SHOULDN'T NEEE THAT^^^
#
# # meta = {'cellid': cellid, 'batch': batch, 'modelname': modelspecname,
# #         'recording': exptid}
#
# xfspec.append(['nems.xforms.init_from_keywords',
#                {'keywordstring': modelspecname, 'meta': meta}])
#
# #xfspec.append(['nems.initializers.rand_phi', {'rand_count': 5}])
# #xfspec.append(['nems.xforms.fit_basic_init', {}])
# #xfspec.append(['nems.xforms.fit_basic', {'tolerance': 1e-6}])
# xfspec.append(['nems.tf.cnnlink_new.fit_tf_init',
#                {'max_iter': 1000, 'early_stopping_tolerance': 5e-4, 'use_modelspec_init': True}])
# xfspec.append(['nems.tf.cnnlink_new.fit_tf',
#                {'max_iter': 1000, 'early_stopping_tolerance': 1e-4, 'use_modelspec_init': True}])
#
# # xfspec.append(['nems.xforms.fit_basic_shrink', {}])
# #xfspec.append(['nems.xforms.fit_basic_cd', {}])
# # xfspec.append(['nems.xforms.fit_iteratively', {}])
#
# #xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])
#
# xfspec.append(['nems.xforms.predict', {}])
# # xfspec.append(['nems.xforms.add_summary_statistics',    {}])
# xfspec.append(['nems.analysis.api.standard_correlation', {},
#                ['est', 'val', 'modelspec', 'rec'], ['modelspec']])
#
# # GENERATE PLOTS
# xfspec.append(['nems.xforms.plot_summary', {}])
#
# # actually do the fit
# log_xf = "NO LOG"
# ctx = {}
# for xfa in xfspec:
#     ctx = xforms.evaluate_step(xfa, ctx)
#################

# initialize context
ctx = {'rec': rec}
meta = {'cellid': 'Classifier', 'batch': 0, 'modelname': modelspecname,
        'recording': 'NAT'}
# xforms logic:
# define ctx dictionary.
# each xforms command takes in ctx and returns updated ctx
ctx.update(xforms.init_from_keywords(keywordstring=modelspecname, meta=meta, **ctx))

# separate out est and val
ctx.update(xforms.mask_for_jackknife(njacks=5, epoch_regex='^STIM_', **ctx))

# make single est/val set, throw away other jackknifes
ctx.update(xforms.jack_subset(keep_only=1, **ctx))

#xfspec.append(['nems.initializers.rand_phi', {'rand_count': 5}])
# ctx.update(xforms.fit_basic_init(**ctx))
# ctx.update(xforms.fit_basic(tolerance=1e-6, **ctx))
ctx.update(cnn.fit_tf_init(max_iter = 1000, early_stopping_tolerance = 5e-4, use_modelspec_init = True,**ctx))
ctx.update(cnn.fit_tf(max_iter = 1000, early_stopping_tolerance=1e-4, use_modelspec_init= True, **ctx))

ctx.update(xforms.predict(**ctx))
ctx.update(xforms.add_summary_statistics(**ctx))

#Make xfspec anyway but don't run it, GUI seems to like it
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
# xfspec.append(['nems.analysis.api.standard_correlation', {},
#                ['est', 'val', 'modelspec', 'rec'], ['modelspec']])
# load in gui
gui.browse_xform_fit(ctx, xfspec)





#This way with no xfspec:
#xfpsec['modelspec'][0] :
#xfpsec['modelspec'][1] : 3 Guassian weight channels
#xfpsec['modelspec'][2] : Describes filter banks, get 1 input, 3 fbs
#xfpsec['modelspec'][3] : Levelshift constant, 3 of them
#[4] : 3 double expenential fit channels


#ONE VERSION OF NEMS I HAD HAD A TYPO IN IT, THIS WAS WORKAROUND
# from nems.initializers import from_keywords
# ctx['modelspec'] = from_keywords(keyword_string=modelspecname,
#                                        meta=meta, registry=None, rec=None,
#                                        input_name='stim',
#                                        output_name='resp')
################################################################