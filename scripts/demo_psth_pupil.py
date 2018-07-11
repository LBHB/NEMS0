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
import nems.uri
from nems import recording
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize

# ----------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.INFO)

nems_dir = os.path.abspath(os.path.dirname(recording.__file__) + '/..')
signals_dir = nems_dir + '/recordings'
modelspecs_dir = nems_dir + '/modelspecs'

cellid = "TAR010c-06-1"
recording_uri = os.path.join(signals_dir, cellid + ".tgz")

# ----------------------------------------------------------------------------
# DATA LOADING

# GOAL: Get your data loaded into memory as a Recording object
logging.info('Loading data...')

# Method #1: Load the data from a local directory
# download demo data if necessary:
recording.get_demo_recordings(signals_dir)
rec = recording.load_recording(recording_uri)

# Method #2: Load the data from baphy using the (incomplete, TODO) HTTP API:
#URL = "http://potoroo:3004/baphy/271/bbl086b-11-1?rasterfs=200"
#rec = Recording.load_url(URL)

# ----------------------------------------------------------------------------
# PREPROCESSING

# create a new signal that will be used to modulate the output of the linear
# predicted response
logging.info('Generating state signal...')
#rec = preproc.make_state_signal(rec, ['active','pupil_bs','pupil_ev'], [''], 'state')
rec = preproc.make_state_signal(rec, ['active','pupil'], [''], 'state')

# mask out data from incorrect trials
rec = preproc.mask_all_but_correct_references(rec)

# calculate a PSTH response for each stimulus, save to a new signal 'psth'
epoch_regex="^STIM_"
rec = preproc.generate_psth_from_resp(rec, epoch_regex, smooth_resp=False)

# ----------------------------------------------------------------------------
# INSPECT THE DATA

resp = rec['resp'].rasterize()
epochs = resp.epochs
epoch_regex="^STIM_"
epoch_list = ep.epoch_names_matching(epochs, epoch_regex)

# list all stimulus events
print(epochs[epochs['name'].isin(epoch_list)])

# list all events of a single stimulus
e = epoch_list[0]
print(epochs[epochs['name'] == e])

# extract raster of all these events on correct or passive trials
# use rec['mask'] to remove all incorrect trial data
raster = resp.extract_epoch(e, mask=rec['mask'])[:,0,:]
t = np.arange(raster.shape[1]) /resp.fs

plt.figure()
plt.subplot(2,1,1)
plt.imshow(raster, interpolation='none', aspect='auto',
           extent=[t[0], t[-1], raster.shape[0], 0])
plt.title('Raster for {}'.format(epoch_list[0]))

plt.subplot(2,1,2)
plt.plot(t, np.nanmean(raster, axis=0))
plt.title('PSTH for {}'.format(epoch_list[0]))

# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC

# GOAL: Define the model that you wish to test

logging.info('Initializing modelspec...')
modelname = 'stategain.S'
meta = {'cellid': cellid, 'modelname': modelname}

# Method #1: create from "shorthand" keyword string
modelspec = nems.initializers.from_keywords(modelname, rec=rec, meta=meta)

modelspecs = [modelspec]

# ----------------------------------------------------------------------------
# DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.
logging.info('Generating jackknife datasets for n-fold cross-validation...')

# create all jackknife sets. the single recording, rec, is now turned into
# lists of recordings for estimation (est) and validation (val). Size of
# signals in each set are the same, but the excluded segments are set to nan.
nfolds = 10
ests, vals, m = preproc.mask_est_val_for_jackknife(rec, modelspecs=None,
                                                   njacks=nfolds)


# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

logging.info('Fitting modelspec(s)...')

modelspecs = nems.analysis.api.fit_nfold(ests, modelspecs,
                                         fitter=scipy_minimize)

# above is shorthand for:
# modelspecs_out=[]
# i=0
# for m,d in zip(modelspecs,ests):
#     i+=1
#     logging.info("Fitting JK {}/{}".format(i,nfolds))
#     modelspecs_out += \
#         nems.analysis.api.fit_basic(d, m, fitter=scipy_minimize)
# modelspecs = modelspecs_out

# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

logging.info('Saving Results...')
ms.save_modelspecs(modelspecs_dir, modelspecs)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

logging.info('Generating summary statistics...')

# generate predictions
ests, vals = nems.analysis.api.generate_prediction(ests, vals, modelspecs)

# evaluate prediction accuracy
modelspecs = nems.analysis.api.standard_correlation(ests, vals, modelspecs)

logging.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspecs[0][0]['meta']['r_fit'],
        modelspecs[0][0]['meta']['r_test']))

# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

logging.info('Generating summary plot...')

# Generate a summary plot
fig = nplt.quickplot({'val': vals, 'modelspecs': modelspecs})
