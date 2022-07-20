# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import logging
import pickle
from pathlib import Path
import gzip
import numpy as np
import pandas as pd

import nems0.analysis.api
import nems0.initializers
import nems0.recording as recording
import nems0.preprocessing as preproc
import nems0.uri
from nems0.fitters.api import scipy_minimize
from nems0.signal import RasterizedSignal

log = logging.getLogger(__name__)

# CONFIGURATION

# figure out data and results paths:
signals_dir = Path(nems0.NEMS_PATH) / 'recordings'
modelspecs_dir = Path(nems0.NEMS_PATH) / 'modelspecs'

# download demo data
recording.get_demo_recordings(signals_dir)

# LOAD AND FORMAT RECORDING DATA
# X (stimulus) is a Frequency X Time matrix, sampled at a rate fs
# Y (response) is a Neuron X Time matrix, also sampled at fs. In this demo,
#   we're analyzing a single neuron, so Y is 1 x T

# this section illustrates several alternative methods for loading,
# each loading from a different file format
load_method = 0

if load_method==0:
    # method 0: load NEMS native recording file
    datafile = signals_dir / 'TAR010c.NAT.fs100.ch18.tgz'
    cellid='TAR010c-18-2'
    rec = recording.load_recording(datafile)
    rec['resp']=rec['resp'].extract_channels([cellid])
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex="^STIM_")
    est=preproc.average_away_epoch_occurrences(est, epoch_regex="^STIM_")
    val=preproc.average_away_epoch_occurrences(val, epoch_regex="^STIM_")

elif load_method==1:
    # method 1: load from a pkl datafile that contains full stim+response data
    # along with metadata (fs, stimulus epoch list)
    datafile = signals_dir / 'TAR010c-18-1.pkl'

    with open(datafile, 'rb') as f:
            #cellid, recname, fs, X, Y, X_val, Y_val = pickle.load(f)
            cellid, recname, fs, X, Y, epochs = pickle.load(f)

    # create NEMS-format recording objects from the raw data
    resp = RasterizedSignal(fs, Y, 'resp', recname, chans=[cellid], epochs=epochs)
    stim = RasterizedSignal(fs, X, 'stim', recname, epochs=epochs)

    # create the recording object from the signals
    signals = {'resp': resp, 'stim': stim}
    rec = recording.Recording(signals)
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex="^STIM_")
    est=preproc.average_away_epoch_occurrences(est, epoch_regex="^STIM_")
    val=preproc.average_away_epoch_occurrences(val, epoch_regex="^STIM_")

elif load_method==2:
    # method 2: load from CSV files - one per response, stimulus, epochs
    # X is a frequency X time spectrgram, sampled at 100 Hz
    # Y is a neuron X time PSTH, aligned with X. Ie, same number of time bins
    # epochs is a list of STIM events with start and stop time of each event
    # in seconds
    # The data have already been averaged across repeats, and the first three
    # stimuli were repeated ~20 times. They will be broken out into the
    # validation recording, used to evaluate model performance. The remaining
    # 90 stimuli will be used for estimation.
    fs=100
    cellid='TAR010c-18-2'
    recname='TAR010c'
    stimfile = signals_dir / 'TAR010c-NAT-stim.csv.gz'
    respfile = signals_dir / 'TAR010c-NAT-resp.csv.gz'
    epochsfile = signals_dir / 'TAR010c-NAT-epochs.csv'

    X=np.loadtxt(gzip.open(stimfile, mode='rb'), delimiter=",", skiprows=0)
    Y=np.loadtxt(gzip.open(respfile, mode='rb'), delimiter=",", skiprows=0)
    # get list of stimuli with start and stop times (in sec)
    epochs = pd.read_csv(epochsfile)

    val_split = 550*3 # validation data are the first 3 5.5 sec stimuli
    resp_chan = 11  # 11th cell is TAR010c-18-2
    X_val = X[:, :val_split]
    X_est = X[:, val_split:]
    epochs_val = epochs.loc[:2]
    epochs_est = epochs.loc[3:]
    Y_val = Y[[resp_chan], :val_split]
    Y_est = Y[[resp_chan], val_split:]

    # create NEMS-format recording objects from the raw data
    resp = RasterizedSignal(fs, Y_est, 'resp', recname, chans=[cellid], epochs=epochs_est)
    stim = RasterizedSignal(fs, X_est, 'stim', recname, epochs=epochs_est)

    # create the recording object from the signals
    signals = {'resp': resp, 'stim': stim}
    est = recording.Recording(signals)

    val_signals = {
            'resp': RasterizedSignal(fs, Y_val, 'resp', recname, chans=[cellid], epochs=epochs_val),
            'stim': RasterizedSignal(fs, X_val, 'stim', recname, epochs=epochs_val)}
    val = recording.Recording(val_signals)


# INITIALIZE MODELSPEC

log.info('Initializing modelspec...')

# Method #1: create from "shorthand" keyword string
#modelspec_name = 'fir.18x15-lvl.1'        # "canonical" linear STRF
#modelspec_name = 'wc.18x1-fir.1x15-lvl.1'        # rank 1 STRF
#modelspec_name = 'wc.18x2.g-fir.2x15-lvl.1'      # rank 2 STRF, Gaussian spectral tuning
modelspec_name = 'wc.18x2.g-fir.2x15-lvl.1-dexp.1'  # rank 2 Gaussian + sigmoid static NL

# record some meta data for display and saving
meta = {'cellid': cellid,
        'batch': 271,
        'modelname': modelspec_name,
        'recording': est.name
        }
modelspec = nems0.initializers.from_keywords(modelspec_name, meta=meta)

# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

log.info('Fitting model ...')

if 'nonlinearity' in modelspec[-1]['fn']:
    # quick fit linear part first to avoid local minima
    modelspec = nems0.initializers.prefit_LN(
            est, modelspec, tolerance=1e-4, max_iter=500)

# then fit full nonlinear model
modelspec = nems0.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

# GENERATE SUMMARY STATISTICS
log.info('Generating summary statistics ...')

# generate predictions
est, val = nems0.analysis.api.generate_prediction(est, val, modelspec)

# evaluate prediction accuracy
modelspec = nems0.analysis.api.standard_correlation(est, val, modelspec)

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

# Generate a summary plot
fig = modelspec.quickplot(rec=val)
fig.show()

# Optional: uncomment to save your figure
# fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# uncomment to browse the validation data
#from nems0.gui.editors import EditorWindow
#ex = EditorWindow(modelspec=modelspec, rec=val)

# TODO SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.
