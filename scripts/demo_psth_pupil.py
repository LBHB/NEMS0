# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import random
import copy

import numpy as np
import matplotlib.pyplot as plt

USE_GUI=True

if USE_GUI:
    import nems.gui.editors as gui
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
from nems.metrics.state import single_state_mod_index

# ----------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.INFO)

nems_dir = os.path.abspath(os.path.dirname(recording.__file__) + '/..')
signals_dir = nems_dir + '/recordings'
modelspecs_dir = nems_dir + '/modelspecs'

cellid = "TAR010c-06-1"
recording_uri = os.path.join(signals_dir, cellid + ".tgz")
#cellid = '951762773' #allen
# ----------------------------------------------------------------------------
# DATA LOADING

# GOAL: Get your data loaded into memory as a Recording object
logging.info('Loading data...')

# Method #1: Load the data from a local directory
# download demo data if necessary:
recording.get_demo_recordings(signals_dir)
rec = recording.load_recording(recording_uri)
#rec=from_nwb_pupil(nwb_filepath,'neuropixel',with_pupil=True,fs=20)
#rec = rec[cellid] #try out allen data

# ----------------------------------------------------------------------------
# PREPROCESSING

# create a new signal that will be used to modulates the output of the linear
# predicted response
logging.info('Generating state signal...')
#rec = preproc.make_state_signal(rec, ['active','pupil'], [''], 'state')
rec = preproc.make_state_signal(rec, state_signals=['pupil'],
                               permute_signals=[''], new_signalname='state')
rec = preproc.make_state_signal(rec, state_signals=['pupil'],
                                permute_signals=['pupil'], new_signalname='state') #shuffled pupil

# mask out data from incorrect trials
#rec = preproc.mask_all_but_correct_references(rec)#not for allen?

# calculate a PSTH response for each stimulus, save to a new signal 'psth'
epoch_regex="^STIM_"
#epoch_regex="^natural_scene" #only natural scene stimulus
rec = preproc.generate_psth_from_resp(rec, epoch_regex=epoch_regex, smooth_resp=False)
#rec=nwb_resp_psth(rec,epoch_regex) #should return similar results as above for allen data

# ----------------------------------------------------------------------------
# INSPECT THE DATA

resp = rec['resp'].rasterize()
epochs = resp.epochs
epoch_regex="^STIM_"
#epoch_regex="^natural_scene" #only natural scene stimulus
epoch_list = ep.epoch_names_matching(epochs, epoch_regex)

# list all stimulus events
print(epochs[epochs['name'].isin(epoch_list)])

# list all events of a single stimulus
epoch = epoch_list[0]
print(epochs[epochs['name'] == epoch])

# extract raster of all these events on correct or passive trials
# use rec['mask'] to remove all incorrect trial data
#raster = resp.extract_epoch(epoch, mask=rec['mask'])[:,0,:]
raster = resp.extract_epoch(epoch)[:,0,:]#no mask for allen
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
modelspec = nems.initializers.from_keywords(modelname, rec=rec, meta=meta,
                                            #input_name='psth')
                                            input_name='psth_sp') #don't subtract spont mean w/allen?

# ----------------------------------------------------------------------------
# DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.
logging.info('Generating jackknife datasets for n-fold cross-validation...')

# create all jackknife sets. the single recording, rec, is now turned into
# lists of recordings for estimation (est) and validation (val). Size of
# signals in each set are the same, but the excluded segments are set to nan.
nfolds = 10
est, val, m = preproc.mask_est_val_for_jackknife(rec, modelspecs=None,
                                                 njacks=nfolds)


# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

logging.info('Fitting modelspec(s)...')

modelspec.tile_jacks(nfolds)
for jack_index, e in enumerate(est.views()):
    logging.info("Fitting JK {}/{}".format(jack_index+1, nfolds))
    modelspec.jack_index = jack_index
    modelspec = nems.analysis.api.fit_basic(e, modelspec, fitter=scipy_minimize)


# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

logging.info('Saving Results...')
ms.save_modelspecs(modelspecs_dir, modelspec.fits())

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

logging.info('Generating summary statistics...')

# generate predictions
est, val = nems.analysis.api.generate_prediction(est, val, modelspec)

# evaluate prediction accuracy
modelspec = nems.analysis.api.standard_correlation(est, val, modelspec)
#slow with allen data
s = nems.metrics.api.state_mod_index(val, epoch='REFERENCE',
                                     psth_name='pred',
                                    state_sig='state', state_chan=[])

modelspec.meta['state_mod'] = s
modelspec.meta['state_chans'] = est['state'].chans

logging.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspec.meta['r_fit'][0,0],
        modelspec.meta['r_test'][0,0]))

print(single_state_mod_index(val, modelspec, epoch='REFERENCE', state_chan="pupil"))

# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

logging.info('Generating summary plot...')

# set modelspec to plot psth for big vs. small pupil
modelspec[0]['plot_fn_idx']=5

# Generate a summary plot

modelspec.quickplot(rec=val,epoch='REFERENCE',include_input=False)
#plot timeseries seperately overides quickplot errors
epoch_bounds = val['resp'].get_epoch_bounds('REFERENCE')
possible_occurrences = np.arange(epoch_bounds.shape[1])
occurrence = possible_occurrences[0]
time_range = epoch_bounds[occurrence]
nems.plots.timeseries.timeseries_from_signals(signals=[val['resp'], val['pred']], channels=0, no_legend=False, 
                                              time_range=time_range, rec=val, sig_name=None)

if USE_GUI:
    # interactive gui
    ex = gui.browse_xform_fit({'modelspec': modelspec, 'val': val}, [])