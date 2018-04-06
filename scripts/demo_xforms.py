# A Template NEMS Script that demonstrates use of xforms for generating
# models that are easy to reload

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
import nems.xforms as xforms

from nems.recording import Recording
from nems.fitters.api import scipy_minimize

# ----------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.INFO)

# figure out data and results paths:
nems_dir = os.path.abspath(os.path.dirname(recording.__file__) + '/..')
signals_dir = nems_dir + '/recordings'
modelspecs_dir = nems_dir + '/modelspecs'

# ----------------------------------------------------------------------------
# DATA LOADING & PRE-PROCESSING
recording_uri = signals_dir + "/TAR010c-18-1.tgz"
recordings = [recording_uri]

xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
          ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
          ['nems.xforms.average_away_stim_occurrences',{}]]

# MODEL SPEC
modelspecname='wcg18x2_fir2x15_lvl1_dexp1'
#modelspecname='wcg18x2_fir2x15_lvl1_dexp1'

meta = {'cellid': 'TAR010c-18-1', 'batch': 271, 'modelname': modelspec_name}

xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])


xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_basic', {}])
xfspec.append(['nems.xforms.predict',    {}])
# xfspec.append(['nems.xforms.add_summary_statistics',    {}])
xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspecs'], ['modelspecs']])

# GENERATE PLOTS
xfspec.append(['nems.xforms.plot_summary',    {}])

# actually do the fit
ctx, log_xf = xforms.evaluate(xfspec)





# ----------------------------------------------------------------------------
# DATA LOADING

# GOAL: Get your data loaded into memory as a Recording object
logging.info('Loading data...')

# Method #1: Load the data from a local directory
# first run download-demo-data to copy down sample file from server
# TODO: make a function that checks and downloads here
rec = Recording.load(signals_dir + "/TAR010c-18-1.tar.gz")

# Method #2: Load the data from baphy using the nems_baphy HTTP API:
#rec = Recording.load("https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/TAR010c-18-1.tar.gz")


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

logging.info('Splitting into estimation and validation data sets...')

# Method #0: Guess which stimuli have the most reps, use those for val
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

# Optional: Take nanmean of ALL occurrences of all signals
est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

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
# very simple linear model
# modelspec = nems.initializers.from_keywords('wc18x2_fir2x15_lvl1')

# Method #1b: constrain spectral tuning to be gaussian, add static output NL
modelspec = nems.initializers.from_keywords('wcg18x2_fir2x15_lvl1_dexp1')

# Method #2: Generate modelspec directly
# TODO: implement this

# record some meta data for display and saving
modelspec[0]['meta']={'cellid': 'TAR010c-18-1', 'batch': 271}


# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

logging.info('Fitting modelspec(s)...')

# Option 1: Use gradient descent on whole data set(Fast)
#modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

# Option 2: quick fit linear part first, then fit full nonlinear model
modelspec = nems.initializers.prefit_to_target(
        est, modelspec, nems.analysis.api.fit_basic,
        target_module='levelshift',
        fitter=scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})
modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

# generate predictions
est, val = nems.analysis.api.generate_prediction(est, val, modelspecs)

# evaluate prediction accuracy
modelspecs = nems.analysis.api.standard_correlation(est, val, modelspecs)


# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# GOAL: Save your results to disk. (BEFORE you screw it up trying to plot!)

# logging.info('Saving Results...')
# ms.save_modelspecs(modelspecs_dir, modelspecs)


# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

logging.info('Generating summary plot...')

# Generate a summary plot
fig = nplt.quickplot({'val': val, 'modelspecs': modelspecs})
#fig.show()

# Optional: Save your figure
#fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)


# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.

# TODO
