# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import random
import copy

import numpy as np
import matplotlib.pyplot as plt

import nems
import nems.initializers
import nems.epoch as ep
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.utils
from nems.recording import Recording
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize

# ----------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.INFO)

relative_signals_dir = '../recordings'
relative_modelspecs_dir = '../modelspecs'

# Convert to absolute paths so they can be passed to functions in
# other directories
signals_dir = os.path.abspath(relative_signals_dir)
modelspecs_dir = os.path.abspath(relative_modelspecs_dir)

# ----------------------------------------------------------------------------
# DATA LOADING

# GOAL: Get your data loaded into memory as a Recording object
logging.info('Loading data...')

# Method #1: Load the data from a local directory
rec = Recording.load(os.path.join(signals_dir, 'TAR010c.NAT.fs50.tgz'))

cellid='TAR010c-16-2'
rec['resp'] = rec['resp'].extract_channels([cellid])

# Method #2: Load the data from baphy using the (incomplete, TODO) HTTP API:
#URL = "http://potoroo:3004/baphy/271/bbl086b-11-1?rasterfs=200"
#rec = Recording.load_url(URL)

logging.info('Generating state signal...')

rec=preproc.make_state_signal(rec,['pupil'],[''],'state')

# ----------------------------------------------------------------------------
# DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.

logging.info('Generating jackknife sets...')

# create all jackknife sets
epoch_name = "REFERENCE"
nfolds=5
est = rec.jackknife_masks_by_epoch(nfolds, epoch_name, tiled=True)
val = rec.jackknife_masks_by_epoch(nfolds, epoch_name, tiled=True, invert=True)
#ests,vals,m=preproc.split_est_val_for_jackknife(rec, modelspecs=None, njacks=nfolds)

# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC

# GOAL: Define the model that you wish to test

logging.info('Initializing modelspec...')

# create from "shorthand" keyword string
modelspec = nems.initializers.from_keywords('wc.18x2.g-fir.2x10-lvl.1-stategain.2')


# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

logging.info('Fitting modelspec(s)...')
modelspecs_out=[]
i=0
for d in est.views():
    m=copy.deepcopy(modelspec)
    i+=1
    logging.info("Fitting JK {}/{}".format(i,nfolds))
    m = nems.initializers.prefit_to_target(
        d, m, nems.analysis.api.fit_basic, 'levelshift',
        fitter=scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}}
        )
    modelspecs_out += \
        nems.analysis.api.fit_basic(d,m,fitter=scipy_minimize)

modelspecs=modelspecs_out

# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

#logging.info('Saving Results...')
#ms.save_modelspecs(modelspecs_dir, modelspecs)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

logging.info('Generating summary statistics...')
modelspecs = nems.analysis.api.standard_correlation(est,val,modelspecs)

logging.info("r_fit={0} r_test={1}".format(modelspecs[0][0]['meta']['r_fit'],
      modelspecs[0][0]['meta']['r_test']))


# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

logging.info('Generating summary plot...')

# Generate a summary plot
fig = nplt.plot_summary(val, modelspecs)
fig.show()


# Optional: Save your figure
#fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# Optional: Load a saved figure programatically as a bytes object
#           that can be used by other python functions
#           (for example, it can be b64 encoded and embedded in a webpage)
#imgbytes = nplt.load_figure_bytes(filepath=fname)

