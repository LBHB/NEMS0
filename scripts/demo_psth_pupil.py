# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import random

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

relative_signals_dir = '../signals'
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
rec = Recording.load(os.path.join(signals_dir, 'eno052d-a1.tgz'))

# Method #2: Load the data from baphy using the (incomplete, TODO) HTTP API:
#URL = "http://potoroo:3004/baphy/271/bbl086b-11-1?rasterfs=200"
#rec = Recording.load_url(URL)

logging.info('Generating state signal...')

rec=preproc.make_state_signal(rec,['pupil'],[],'state')

# ----------------------------------------------------------------------------
# DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.


logging.info('Withholding validation set data...')

# Method #0: Try to guess which stimuli have the most reps, use those for val
est = rec.jackknife_by_time(10, 1, invert=False, excise=False)
val = rec.jackknife_by_time(10, 1, invert=True, excise=False)

epoch_regex='^STIM_'
resp_est=est['resp']
resp_val=val['resp']

epochs_to_extract = ep.epoch_names_matching(resp_est.epochs, epoch_regex)
folded_matrices = resp_est.extract_epochs(epochs_to_extract)

# 2. Average over all reps of each stim and save into dict called psth.
per_stim_psth = dict()
for k in folded_matrices.keys():
    per_stim_psth[k] = np.nanmean(folded_matrices[k], axis=0)

# 3. Invert the folding to unwrap the psth into a predicted spike_dict by
#   replacing all epochs in the signal with their average (psth)
respavg_est = resp_est.replace_epochs(per_stim_psth)
respavg_est.name = 'stim'
est.add_signal(respavg_est)

respavg_val = resp_est.replace_epochs(per_stim_psth)
respavg_val.name = 'stim'
val.add_signal(respavg_val)



# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC

# GOAL: Define the model that you wish to test

logging.info('Initializing modelspec(s)...')

# Method #1: create from "shorthand" keyword string
#modelspec = nems.initializers.from_keywords('wcg18x1_fir15x1_lvl1_dexp1')
modelspec = nems.initializers.from_keywords('stategain2')

#modelspec = nems.initializers.from_keywords('wcg18x2_fir15x2_lvl1_stategain2')

# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

logging.info('Fitting modelspec(s)...')

# Option 1: Use gradient descent on whole data set(Fast)
modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)


# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS


logging.info('Saving Results...')
ms.save_modelspecs(modelspecs_dir, modelspecs)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

logging.info('Generating summary statistics...')

new_rec = [ms.evaluate(val, m) for m in modelspecs]
r_test = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_rec]
new_rec = [ms.evaluate(est, m) for m in modelspecs]
r_fit = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_rec]
modelspecs[0][0]['meta']['r_fit']=np.mean(r_fit)
modelspecs[0][0]['meta']['r_test']=np.mean(r_test)

logging.info("r_fit={0} r_test={1}".format(modelspecs[0][0]['meta']['r_fit'],
      modelspecs[0][0]['meta']['r_test']))


# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

logging.info('Generating summary plot...')

# Generate a summary plot
plt.figure();
plt.plot(val['pred'].as_continuous().T)
plt.plot(val['state'].as_continuous().T/100)


# Optional: Save your figure
#fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# Optional: Load a saved figure programatically as a bytes object
#           that can be used by other python functions
#           (for example, it can be b64 encoded and embedded in a webpage)
#imgbytes = nplt.load_figure_bytes(filepath=fname)

