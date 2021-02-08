# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:07:19 2021

@author: vlab
"""

# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import logging
import pickle
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")

import nems.analysis.api
import nems.initializers
import nems.recording as recording
import nems.preprocessing as preproc
import nems.uri
from nems.fitters.api import scipy_minimize
#from nems.tf.cnnlink_new import fit_tf, fit_tf_init

from nems.signal import RasterizedSignal

log = logging.getLogger(__name__)

# CONFIGURATION

# figure out data and results paths:
signals_dir = Path(nems.NEMS_PATH) / 'recordings'
modelspecs_dir = Path(nems.NEMS_PATH) / 'modelspecs'

# LOAD AND FORMAT RECORDING DATA
# X (stimulus) is a Frequency X Time matrix, sampled at a rate fs
# Y (response) is a Neuron X Time matrix, also sampled at fs. In this demo,
#   we're analyzing a single neuron, so Y is 1 x T


# method 2: load from CSV files - one per response, stimulus, epochs
# X is a frequency X time spectrgram, sampled at 100 Hz
# Y is a neuron X time PSTH, aligned with X. Ie, same number of time bins
# epochs is a list of STIM events with start and stop time of each event
# in seconds
# The data have already been averaged across repeats, and the first three
# stimuli were repeated ~20 times. They will be broken out into the
# validation recording, used to evaluate model performance. The remaining
# 90 stimuli will be used for estimation.
fs=50
cellid='MS_u0004_f0025'
recname='MS_u0004'
stimfile = signals_dir / 'MS_u0004_f0025_stim.csv.gz'
respfile = signals_dir / 'MS_u0004_f0025_resp.csv.gz'
epochsfile = signals_dir / 'MS_u0004_f0025_epochs.csv'

X=np.loadtxt(gzip.open(stimfile, mode='rb'), delimiter=",", skiprows=0)
Y=np.loadtxt(gzip.open(respfile, mode='rb'), delimiter=",", skiprows=0)
# get list of stimuli with start and stop times (in sec)
epochs = pd.read_csv(epochsfile)

# create NEMS-format recording objects from the raw data
resp = RasterizedSignal(fs, Y, 'resp', recname, chans=[cellid], epochs=epochs.loc[:])
stim = RasterizedSignal(fs, X, 'stim', recname, epochs=epochs.loc[:])

# create the recording object from the signals
signals = {'resp': resp, 'stim': stim}
rec = recording.Recording(signals)

#generate est/val set_sets
#nfolds=10
#est = rec.jackknife_masks_by_time(njacks=nfolds, invert=False) #VATSUN - doesnt work
#val = rec.jackknife_masks_by_time(njacks=nfolds, invert=True)

est, val = rec.split_at_time(fraction=0.1) # VATSUN: Fraction=0.1 I think specifies the validation set

# INITIALIZE MODELSPEC

log.info('Initializing modelspec...')

# Method #1: create from "shorthand" keyword string
modelspec_name = 'fir.18x10-lvl.1-dexp.1'        # "canonical" linear STRF + nonlinearity
#modelspec_name = 'fir.18x19.nc9-lvl.1-dexp.1'     # "canonical" linear STRF + nonlinearity + anticausal time lags

#modelspec_name = 'wc.18x1-fir.1x10-lvl.1'        # rank 1 STRF
#modelspec_name = 'wc.18x2.g-fir.2x10-lvl.1'      # rank 2 STRF, Gaussian spectral tuning
#modelspec_name = 'wc.18x2.g-fir.2x10-lvl.1-dexp.1'  # rank 2 Gaussian + sigmoid static NL

# record some meta data for display and saving
meta = {'cellid': cellid,
        'batch': 271,
        'modelname': modelspec_name,
        'recording': est.name
        }
modelspec = nems.initializers.from_keywords(modelspec_name, meta=meta)
if est.view_count>1:
    log.info('Tiling modelspec to match number of est views')
    modelspec.tile_jacks(nfolds)

# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

log.info('Fitting model ...')

for jack_idx, e in enumerate(est.views()):
    modelspec.jack_index = jack_idx
    log.info("----------------------------------------------------")
    log.info("Fitting: fold %d/%d", jack_idx + 1, modelspec.jack_count)

    if 'nonlinearity' in modelspec[-1]['fn']:
        # quick fit linear part first to avoid local minima
        modelspec = nems.initializers.prefit_LN(
                est, modelspec, tolerance=1e-4, max_iter=500) #VATSUN- doesnt work if you change max_iter to 2000

        # uncomment to try TensorFlow pre-fitter:
        #modelspec = fit_tf_init(modelspec, est, epoch_name=None)['modelspec']

    # then fit full nonlinear model -- scipy fitter
    modelspec = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

    # uncomment to try TensorFlow fitter:
    #modelspec = fit_tf(modelspec, est, epoch_name=None)['modelspec']

# GENERATE SUMMARY STATISTICS
log.info('Generating summary statistics ...')

# generate predictions
est, val = nems.analysis.api.generate_prediction(est, val, modelspec)

# evaluate prediction accuracy
modelspec = nems.analysis.api.standard_correlation(est, val, modelspec)

log.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspec.meta['r_fit'][0][0],
        modelspec.meta['r_test'][0][0]))

# SAVE YOUR RESULTS

# uncomment to save model to disk
# logging.info('Saving Results...')
# modelspec.save_modelspecs(modelspecs_dir, modelspecs)

# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

log.info('Generating summary plot ...')

# uncomment to browse the validation data
#from nems.gui.editors import EditorWindow
#ex = EditorWindow(modelspec=modelspec, rec=val)

# Generate a summary plot
fig = modelspec.quickplot(rec=val)
fig.show()

fig = modelspec.quickplot(rec=est) # VATSUN: added to determine what fraction split corresponded to
fig.show()

# Optional: uncomment to save your figure
# fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# TODO SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.
