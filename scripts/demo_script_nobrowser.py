# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import pandas as pd
import pickle
#from nems.gui.recording_browser import browse_recording, browse_context
import nems
import nems.initializers
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.utils
import nems.uri
import nems.recording as recording
from nems.signal import RasterizedSignal
from nems.fitters.api import scipy_minimize

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIGURATION


# figure out data and results paths:
signals_dir = nems.NEMS_PATH + '/recordings'
modelspecs_dir = nems.NEMS_PATH + '/modelspecs'
recording.get_demo_recordings(signals_dir)

pkl_file=signals_dir + "/TAR010c-18-1.pkl"

# ----------------------------------------------------------------------------
# LOAD AND FORMAT RECORDING DATA

with open(pkl_file, 'rb') as f:  # Python 3: open(..., 'rb')
    cellid, recname, fs, X_est, Y_est, X_val, Y_val = pickle.load(f)

epochs = None
stimchans = [str(x) for x in range(X_est.shape[0])]
# borrowed from recording.load_recording_from_arrays

# est recording - for model fitting
resp = RasterizedSignal(fs, Y_est, 'resp', recname, chans=[cellid])
stim = RasterizedSignal(fs, X_est, 'stim', recname, chans=stimchans)
signals = {'resp': resp, 'stim': stim}
est = recording.Recording(signals)

# val recording - for testing predictions
resp = RasterizedSignal(fs, Y_val, 'resp', recname, chans=[cellid])
stim = RasterizedSignal(fs, X_val, 'stim', recname, chans=stimchans)
signals = {'resp': resp, 'stim': stim}
val = recording.Recording(signals)


# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC
#
# GOAL: Define the model that you wish to test

log.info('Initializing modelspec(s)...')

# Method #1: create from "shorthand" keyword string
# very simple linear model
modelspec_name='wc.18x2.g-fir.2x15-lvl.1'

# Method #1b: constrain spectral tuning to be gaussian, add static output NL
#modelspec_name='wc.18x2.g-fir.2x15-lvl.1-dexp.1'

# record some meta data for display and saving
meta = {'cellid': cellid, 'batch': 271,
        'modelname': modelspec_name, 'recording': cellid}
modelspec = nems.initializers.from_keywords(modelspec_name, meta=meta)



# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

log.info('Fitting modelspec(s)...')

# quick fit linear part first to avoid local minima
modelspec = nems.initializers.prefit_to_target(
        est, modelspec, nems.analysis.api.fit_basic,
        target_module='levelshift',
        fitter=scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})


# then fit full nonlinear model
modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

log.info('Generating summary statistics...')

# generate predictions
est, val = nems.analysis.api.generate_prediction(est, val, modelspecs)

# evaluate prediction accuracy
modelspecs = nems.analysis.api.standard_correlation(est, val, modelspecs)

log.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspecs[0][0]['meta']['r_fit'][0],
        modelspecs[0][0]['meta']['r_test'][0]))

# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# uncomment to save model to disk:

# logging.info('Saving Results...')
# ms.save_modelspecs(modelspecs_dir, modelspecs)


# ----------------------------------------------------------------------------
# GENERATE PLOTS
#
# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

log.info('Generating summary plot...')

# Generate a summary plot
context = {'val': val, 'modelspecs': modelspecs, 'est': est}
fig = nplt.quickplot(context)
fig.show()

# Optional: uncomment to save your figure
# fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# browse the validation data
#aw = browse_recording(val[0], signals=['stim', 'pred', 'resp'], cellid=cellid)


# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.

# TODO
