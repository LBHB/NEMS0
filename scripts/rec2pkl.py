# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
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

# ----------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.INFO)

# figure out data and results paths:
nems_dir = os.path.abspath(os.path.dirname(recording.__file__) + '/..')
signals_dir = nems_dir + '/recordings'
modelspecs_dir = nems_dir + '/modelspecs'

# ----------------------------------------------------------------------------
# DATA LOADING
#
# GOAL: Get your data loaded into memory as a Recording object

logging.info('Loading data...')

# Method #1: Load the data from a local directory
# download demo data if necessary:
recording.get_demo_recordings(signals_dir, name="TAR010c-18-1.tgz")

# load into a recording object
rec = recording.load_recording(signals_dir + "/TAR010c-18-1.tgz")

# ----------------------------------------------------------------------------
# DATA PREPROCESSING
#
# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.

if True:
    # aside - generate datasets from scratch
    X = rec['stim'].as_continuous()
    Y = rec['resp'].as_continuous()

    fs = rec['resp'].fs
    recname = 'TAR010c'
    cellid = "TAR010c-18-1"
    epochs = rec['resp'].epochs

    pkl_file = signals_dir + "/TAR010c-18-1.pkl"

    with open(pkl_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([cellid, recname, fs, X, Y, epochs], f)

    # Getting back the objects:
    with open(pkl_file, 'rb') as f:  # Python 3: open(..., 'rb')
        cellid, recname, fs, X, Y, epochs = pickle.load(f)


else:
    logging.info('Splitting into estimation and validation data sets...')

    # Method #1: Find which stimuli have the most reps, use those for val
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

    # Optional: Take nanmean of ALL occurrences of all signals
    est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_').apply_mask()
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_').apply_mask()

    # aside - generate datasets from scratch
    X_est = est['stim'].as_continuous()
    Y_est = est['resp'].as_continuous()
    X_val = val['stim'].as_continuous()
    Y_val = val['resp'].as_continuous()

    fs = est['resp'].fs
    recname = 'TAR010c'
    cellid="TAR010c-18-1"

    pkl_file=signals_dir + "/TAR010c-18-1.pkl"

    with open(pkl_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([cellid, recname, fs, X_est, Y_est, X_val, Y_val], f)



    # Getting back the objects:
    with open(pkl_file, 'rb') as f:  # Python 3: open(..., 'rb')
        cellid, recname, fs, X_est, Y_est, X_val, Y_val = pickle.load(f)




    epochs = est['resp'].epochs
    stimchans = [str(x) for x in range(X_est.shape[0])]
    # borrowed from recording.load_recording_from_arrays

    # est recording
    resp = RasterizedSignal(fs, Y_est, 'resp', recname, chans=[cellid])
    stim = RasterizedSignal(fs, X_est, 'stim', recname, chans=stimchans)
    signals = {'resp': resp, 'stim': stim}
    est = recording.Recording(signals)

    # val recording
    resp = RasterizedSignal(fs, Y_val, 'resp', recname, chans=[cellid])
    stim = RasterizedSignal(fs, X_val, 'stim', recname, chans=stimchans)
    signals = {'resp': resp, 'stim': stim}
    val = recording.Recording(signals)


    browse_recording(est, signals=['stim', 'resp'], cellid=cellid)



