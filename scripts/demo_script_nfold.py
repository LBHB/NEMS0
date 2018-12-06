# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import pandas as pd
import pickle

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
from nems.gui.recording_browser import browse_recording, browse_context

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIGURATION


# figure out data and results paths:
signals_dir = nems.NEMS_PATH + '/recordings'
modelspecs_dir = nems.NEMS_PATH + '/modelspecs'
recording.get_demo_recordings(signals_dir)

datafile = signals_dir + "/TAR010c-18-1.pkl"

# ----------------------------------------------------------------------------
# LOAD AND FORMAT RECORDING DATA

with open(datafile, 'rb') as f:
        cellid, recname, fs, X, Y, epochs = pickle.load(f)

stimchans = [str(x) for x in range(X.shape[0])]
# borrowed from recording.load_recording_from_arrays

resp = RasterizedSignal(fs, Y, 'resp', recname, epochs=epochs, chans=[cellid])
stim = RasterizedSignal(fs, X, 'stim', recname, epochs=epochs, chans=stimchans)
signals = {'resp': resp, 'stim': stim}
rec = recording.Recording(signals)

epoch_name = "REFERENCE"
nfolds=5
est = rec.jackknife_masks_by_epoch(nfolds, epoch_name, tiled=True)
val = rec.jackknife_masks_by_epoch(nfolds, epoch_name, tiled=True, invert=True)

#est, val = rec.split_at_time(0.2)
#est, val = rec.split_using_epoch_occurrence_counts(epoch_regex="^STIM_")
#est = preproc.average_away_epoch_occurrences(est, epoch_regex="^STIM_")
#val = preproc.average_away_epoch_occurrences(val, epoch_regex="^STIM_")

# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC
#
# GOAL: Define the model that you wish to test

log.info('Initializing modelspec(s)...')

# Method #1: create from "shorthand" keyword string
# very simple linear model
modelspec_name='wc.18x1.g-fir.1x15-lvl.1'
#modelspec_name='wc.18x2.g-fir.2x15-lvl.1'

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
modelspec.tile_fits(nfolds)
for m, e in zip(modelspec.fits(), est.views()):
    m = nems.initializers.prefit_to_target(
        e, m, nems.analysis.api.fit_basic,
        target_module='levelshift',
        fitter=scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})

#modelspecs = [nems.initializers.prefit_to_target(
#        e, modelspec, nems.analysis.api.fit_basic,
#        target_module='levelshift',
#        fitter=scipy_minimize,
#        fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})
#        for e in est.views()]


# then fit full nonlinear model
#modelspecs = [nems.analysis.api.fit_basic(e, m, fitter=scipy_minimize)[0]
#              for m, e in zip(modelspecs, est.views())]
for fit_index, e in enumerate(est.views()):
    logging.info("Fitting JK {}/{}".format(fit_index+1, nfolds))
    modelspec.fit_index = fit_index
    modelspec = nems.analysis.api.fit_basic(e, modelspec, fitter=scipy_minimize)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

log.info('Generating summary statistics...')

# generate predictions
est, val = nems.analysis.api.generate_prediction(est, val, modelspec)

# evaluate prediction accuracy
modelspec = nems.analysis.api.standard_correlation(est, val, modelspec)

log.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspec.meta['r_fit'][0],
        modelspec.meta['r_test'][0]))

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
context = {'val': val, 'modelspecs': modelspec.fits(), 'est': est}
fig = nplt.quickplot(context)
fig.show()

# Optional: uncomment to save your figure
# fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# browse the validation data
#aw = browse_recording(val, signals=['stim', 'pred', 'resp'], cellid=cellid)



# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.

# TODO
