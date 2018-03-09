import os
import random
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import nems.epoch as ep
import nems.modelspec as ms
import nems.plots.api as nplt
from nems.recording import Recording

# specify directories for loading data and fitted modelspec
signals_dir = '../signals'
#signals_dir = '/home/jacob/auto/data/batch271_fs100_ozgf18/'
modelspecs_dir = '../modelspecs'

# load the data
rec = Recording.load(os.path.join(signals_dir, 'TAR010c-18-1'))

# Add a new signal, respavg, to the recording, in 4 steps

# 1. Fold matrix over all stimuli, returning a dictionary where keys are stimuli
#    and each value in the dictionary is (reps X cell X bins)
epochs_to_extract = ep.epoch_names_matching(rec.epochs, '^STIM_')
folded_matrix = rec['resp'].extract_epochs(epochs_to_extract)

# 2. Average over all reps of each stim and save into dict called psth.
per_stim_psth = dict()
for k in folded_matrix.keys():
    per_stim_psth[k] = np.nanmean(folded_matrix[k], axis=0)

# 3. Invert the folding to unwrap the psth back out into a predicted spike_dict by
# simply replacing all epochs in the signal with their psth
respavg = rec['resp'].replace_epochs(per_stim_psth)
respavg.name = 'respavg'

# 4. Now add the signal to the recording
rec.add_signal(respavg)

# Now split into est and val data sets
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
# est, val = rec.split_at_time(0.8)

# Load some modelspecs and create their predictions
modelspecs = ms.load_modelspecs(modelspecs_dir, 'TAR010c-18-1')#, regex=('^TAR010c-18-1\.{\d+}\.json'))
# Testing summary statistics:
means, stds = ms.summary_stats(modelspecs)
print("means: {}".format(means))
print("stds: {}".format(stds))

pred = [ms.evaluate(val, m)['pred'] for m in modelspecs]

# Shorthands for unchanging signals
stim = val['stim']
resp = val['resp']
respavg = val['respavg']


def plot_layout(plot_fn_struct):
    '''
    Accepts a list of lists of functions of 1 argument (ax).
    Basically a fancy subplot that lets you lay out functions without
    worrying about details. See example below
    '''
    # Count how many plot functions we want
    nrows = len(plot_fn_struct)
    ncols = max([len(row) for row in plot_fn_struct])
    # Set up the subplots
    fig = plt.figure(figsize=(12,12))
    for r, row in enumerate(plot_fn_struct):
        for c, plotfn in enumerate(row):
            colspan = max(1, int(ncols / len(row)))
            ax = plt.subplot2grid((nrows, ncols), (r, c), colspan=colspan)
            plotfn(ax=ax)
    return fig

# Example of how to plot a complicated thing:
def my_scatter(ax): nplt.plot_scatter_smoothed(resp, pred[0], ax=ax, title=rec.name)
def my_spectro(ax): nplt.spectrogram_from_epoch(stim, 'TRIAL', ax=ax, occurrence=2)
sigs = [respavg]
sigs.extend(pred)
sigs = nplt.combine_signal_channels(sigs, 1, None)
occurrences = nplt.pad_to_signals(sigs, 0)
# use the 0th channel for respavg and 1st channel for concatenated pred
# (just to test that channels were concatenated properly)
channels = nplt.pad_to_signals(sigs, [0, 1])
def my_timeseries(ax) : nplt.timeseries_from_epoch(sigs, 'TRIAL', ax=ax,
                                                   occurrences=occurrences,
                                                   channels=channels)
def my_histogram(ax) : nplt.pred_error_hist(resp, pred[0], ax=ax)
fig = plot_layout([[my_scatter, my_scatter, my_scatter],
                   [my_spectro],
                   [my_timeseries],
                   [my_histogram]])

fig.tight_layout()
fig.show()

# Pause here before quitting
plt.show()

#exit()

################################################################################
# TODO: How to ensure that est, val split are the same as they
#       were for when the modelspec was fitted?
#       Will this information be in the modelspec metadata?
#       Sometihng like meta: {'segmentor': ('split_at_time', 0.8)}?

# TODO: Fix problem with splitting messing up epochs. This was workaround:
#stim.epochs = stim.trial_epochs_from_occurrences(occurrences=377)
#resp.epochs = resp.trial_epochs_from_occurrences(occurrences=377)

# TODO: How do we view multiple occurrences of trials?
#       etc present
#       at once.

# use defaults for all plot functions using the 'high-level' plotting functions
#plot_fns = [nplt.pred_vs_act_scatter, nplt.pred_vs_act_psth]
#frozen_fns = nplt.freeze_defaults(plot_fns, val, loaded_modelspecs[0],
#                                  ms.evaluate)
#fig = nplt.simple_grid(frozen_fns, nrows=len(plot_fns))
#print("Signals with all epochs included")
#fig.show()

