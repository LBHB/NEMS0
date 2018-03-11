# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import nems
import nems.initializers
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.utils
import nems.uri
from nems.recording import Recording
from nems.fitters.api import scipy_minimize

# ----------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.INFO)

relative_signals_dir = '../signals'
#relative_signals_dir = '/home/jacob/auto/data/batch271_fs100_ozgf18/'
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
# rec = Recording.load("/auto/data/recordings/TAR010c-18-1")
# rec = Recording.load("file:///auto/data/recordings/TAR010c-18-1")

# Method #2: Load the data from baphy using the nems_baphy HTTP API:
# rec = Recording.load("http://potoroo/recordings/TAR010c-18-1.tar.gz")
rec = Recording.load("http://potoroo/baphy/271/bbl086b-11-1")

# Method #3: Load the data from S3: (TODO)
# stimfile=("https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"
#           +cellid+"_NAT_stim_ozgf_c18_fs100.mat")
# respfile=("https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"
#           +cellid+"_NAT_resp_fs100.mat")
# rec = lbhb.fetch_signals_over_http(stimfile, respfile)

# Method #4: Create a Recording object from an array, manually

# need a list of array-like data structures
#arrays = [x, y, z]
# a name for the recording that will hold the signals
#rec_name = 'my_recording'
# the sampling rate for the signals, or a list of
# individual sampling rates (if different)
#fs = [100, 100, 200]
# a list of signal names (optional, but preferred)
#names = ['stim', 'resp', 'reference']
# a list of keyword arguments for each signal,
# such as channel names or epochs (also optional)
#kwargs = [{'chans': ['2kHz', '4kHz', '8kHz']},
#          {'chans': ['spike_rate']},
#          {'meta': {'experiment': 'oddball_2'}}]
#rec = Recording.load_from_arrays(arrays, rec_name, fs, sig_names=names,
#                                 signal_kwargs = kwargs)


# ----------------------------------------------------------------------------
# OPTIONAL PREPROCESSING
logging.info('Preprocessing data...')

# Add a respavg signal to the recording now, so we don't have to do it later
# on both the est and val sets seperately.
rec = preproc.add_average_sig(rec, signal_to_average='resp',
                              new_signalname='resp', # NOTE: ADDING AS RESP NOT RESPAVG FOR TESTING
                              epoch_regex='^STIM_')

# ----------------------------------------------------------------------------
# DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.

logging.info('Withholding validation set data...')

# Method #0: Try to guess which stimuli have the most reps, use those for val
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

# Optional: Take nanmean of ALL occurrences of all signals
# est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
# val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

# Method #1: Split based on time, where the first 80% is estimation data and
#            the last, last 20% is validation data.
# est, val = rec.split_at_time(0.8)

# Method #2: Split based on repetition number, rounded to the nearest rep.
# est, val = rec.split_at_rep(0.8)

# Method #3: Use the whole data set! (Usually for doing n-fold cross-val)
# est = rec
# val = rec


# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC

# GOAL: Define the model that you wish to test

logging.info('Initializing modelspec(s)...')

# Method #1: create from "shorthand" keyword string
modelspec = nems.initializers.from_keywords('wc18x1_lvl1_fir15x1_dexp1')
# modelspec = nems.initializers.from_keywords('wc18x1_lvl1_fir15x1_logsig1')
# modelspec = nems.initializers.from_keywords('wc18x1_lvl1_fir15x1_qsig1')
# modelspec = nems.initializers.from_keywords('wc18x1_lvl1_fir15x1_tanh1')

# Method #2: Load modelspec(s) from disk
# TODO: allow selection of a specific modelspec instead of ALL models for this data!!!!
# modelspecs = ms.load_modelspecs(modelspecs_dir, 'TAR010c-18-1')

# Optional: start from some prior
# modelspec = nems.priors.set_random_phi(modelspec)

# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

logging.info('Fitting modelspec(s)...')

# Option 1: Use gradient descent on whole data set(Fast)
modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

# Option 2: Split the est data into 10 pieces, fit them, and average
# modelspecs = nems.analysis.api.fit_random_subsets(est, modelspec, nsplits=10)
# result = average(modelspecs...)

# Option 3: Fit 4 jackknifes of the data, and return all of them.
# modelspecs = nems.analysis.api.fit_jackknifes(est, modelspec, njacks=4)

# Option 4: Divide estimation data into 10 subsets; fit all sets separately
# modelspecs = nems.analysis.api.fit_subsets(est, modelspec, nsplits=3)

# Option 5: Start from random starting points 4 times
#modelspecs = nems.analysis.api.fit_from_priors(est, modelspec, ntimes=4)

# TODO: Perturb around the modelspec to get confidence intervals

# TODO: Use simulated annealing (Slow, arguably gets stuck less often)
# modelspecs = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.annealing)

# TODO: Use Metropolis algorithm (Very slow, gives confidence interval)
# modelspecs = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.metropolis)

# TODO: Use 10-fold cross-validated evaluation
# fitter = partial(nems.cross_validator.cross_validate_wrapper, gradient_descent, 10)
# modelspecs = nems.analysis.fit_cv(est, modelspec, folds=10)


# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# GOAL: Save your results to disk. (BEFORE you screw it up trying to plot!)

logging.info('Saving Results...')

ms.save_modelspecs(modelspecs_dir, modelspecs)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

logging.info('Generating summary statistics...')

# TODO

# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

logging.info('Generating summary plot...')

# Generate a summary plot
fig = nplt.plot_summary(val, modelspecs)
fig.show()

# Optional: See how well your best result predicts the validation data set
# nems.plot.predictions(val, [results[0]]) # TODO

# Optional: See how all the results predicted
# nems.plot.predictions(val, results) # TODO

# Optional: Compute the confidence intervals on your results
# nems.plot.confidence_intervals(val, results) # TODO

# Optional: View the prediction of the best result according to MSE
# nems.plot.best_estimator(val, results, metric=nems.metrics.mse) # TODO

# Optional: View the posterior parameter probability distributions
# nems.plot.posterior(val, results) # TODO

# Pause before quitting

# Optional: Save your figure
fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# Optional: Load a saved figure programatically as a bytes object
#           that can be used by other python functions
#           (for example, it can be b64 encoded and embedded in a webpage)
imgbytes = nplt.load_figure_bytes(filepath=fname)

# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.

# TODO
