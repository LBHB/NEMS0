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
import matplotlib.pyplot as plt

import nems.analysis.api
import nems.initializers
import nems.recording as recording
import nems.preprocessing as preproc
import nems.uri
from nems import xforms, xform_helper
from nems.fitters.api import scipy_minimize
#from nems.tf.cnnlink_new import fit_tf, fit_tf_init

from nems.signal import RasterizedSignal

log = logging.getLogger(__name__)

# CONFIGURATION

# figure out data and results paths:
signals_dir = Path(nems.NEMS_PATH) / 'recordings'
modelspecs_dir = Path(nems.NEMS_PATH) / 'modelspecs'

# LOAD AND FORMAT RECORDING DATA
from nems.plugins.loaders import load_sadagopan
fs=50
cellid='MS_u0004_f0025'
recname='MS_u0004'
batch = 1  # integer identifier for analysis group
load_command = 'nems.plugins.loaders.load_sadagopan'

# Method #1: create from "shorthand" keyword string
#modelspec_name = 'fir.18x10-lvl.1-dexp.1'        # "canonical" linear STRF + nonlinearity
#modelspec_name = 'fir.18x19.nc9-lvl.1-dexp.1'     # "canonical" linear STRF + nonlinearity + anticausal time lags

modelspec_name = 'wc.18x1-fir.1x10-lvl.1'        # rank 1 STRF
#modelspec_name = 'wc.18x2.g-fir.2x10-lvl.1'      # rank 2 STRF, Gaussian spectral tuning
#modelspec_name = 'wc.18x2.g-fir.2x10-lvl.1-dexp.1'  # rank 2 Gaussian + sigmoid static NL

modelname = "vsload.fs50-tev_" + modelspec_name + "_init-basic"
#xfspec, ctx = xform_helper.fit_model_xform(cellid, batch, modelname, saveInDB=False,
#                              returnModel=True)

# generate modelspec
xfspec = []

# load from external format
#rec = load_sadagopan(cellid=cellid, recname=recname, fs=fs)['rec']
xfspec.append([load_command,
               {'cellid': cellid, 'recname': recname, 'fs': fs}])

# uncomment to average across
#xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])


# generate est/val sets
#xfspec.append(['nems.xforms.split_at_time', {'valfrac': 0.1}])

meta = {'cellid': cellid, 'batch': batch, 'modelname': modelspec_name,
        'recording': recname}
xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspec_name, 'meta': meta}])

xfspec.append(['nems.xforms.mask_for_jackknife',{'njacks':10,'by_time':True}])

# generate 5 random initial conditions (instead of just 1 by default)
#xfspec.append(['nems.initializers.rand_phi', {'rand_count': 5}])

# scipy fitter
#xfspec.append(['nems.xforms.fit_basic_init', {}])
#xfspec.append(['nems.xforms.fit_basic', {'tolerance': 1e-6}])

# TF fitter
xfspec.append(['nems.tf.cnnlink_new.fit_tf_init',
               {'max_iter': 1000, 'early_stopping_tolerance': 5e-4, 'use_modelspec_init': True, 'epoch_name': ""}])
xfspec.append(['nems.tf.cnnlink_new.fit_tf',
               {'max_iter': 1000, 'early_stopping_tolerance': 1e-4, 'use_modelspec_init': True, 'epoch_name': ""}])

# if called rand_phi, use this to pick the best fit
#xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

xfspec.append(['nems.xforms.predict', {}])
xfspec.append(['nems.xforms.add_summary_statistics',    {}])
#xfspec.append(['nems.analysis.api.standard_correlation', {},
#               ['est', 'val', 'modelspec', 'rec'], ['modelspec']])

# GENERATE PLOTS
xfspec.append(['nems.xforms.plot_summary', {}])

# Run the xforms script
log_xf = "NO LOG"
ctx = {}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)
# or just call the xforms wrapper
#ctx, log_xf = xforms.evaluate(xfspec)

modelspec=ctx['modelspec']

log.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspec.meta['r_fit'][0][0],
        modelspec.meta['r_test'][0][0]))

# SAVE YOUR RESULTS
#save_data = xforms.save_analysis(modelspec.meta['modelpath'],
#                                 recording=ctx['rec'],
#                                 modelspec=modelspec,
#                                 xfspec=xfspec,
#                                 figures=ctx['figures'],
#                                 log=log_xf,
#                                 update_meta=False)

# uncomment to save model to disk
# logging.info('Saving Results...')
# modelspec.save_modelspecs(modelspecs_dir, modelspecs)

# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

# uncomment to browse the validation data
#from nems.gui.editors import EditorWindow
#ex = EditorWindow(modelspec=modelspec, rec=val)

# Generate a summary plot
#fig = modelspec.quickplot(rec=ctx['val'])
#fig.show()

#fig = modelspec.quickplot(rec=ctx['est']) # VATSUN: added to determine what fraction split corresponded to
#fig.show()

# Optional: uncomment to save your figure
# fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

plt.show()
