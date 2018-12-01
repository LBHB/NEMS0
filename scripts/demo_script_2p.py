# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import pandas as pd
import pickle
import sys
import numpy as np

from nems.gui.recording_browser import browse_recording, browse_context
import nems
import nems.initializers
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.xforms as xforms
import nems.utils
import nems.uri
import nems.recording as recording
from nems.signal import RasterizedSignal
from nems.fitters.api import scipy_minimize


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# LOAD AND FORMAT RECORDING DATA

# 2p data from Polley Lab at EPL
respfile=nems.get_setting('NEMS_RECORDINGS_DIR') + '/data_nems_2p/neurons.csv'
stimfile=nems.get_setting('NEMS_RECORDINGS_DIR') + '/data_nems_2p/stim_spectrogram.csv'
exptid = "POL001"
cellid = "POL001-080"
batch = None
load_command='nems.demo.loaders.load_polley_data'

xfspec = []
xfspec.append(['nems.xforms.load_recording_wrapper', {'load_command': load_command}])
xfspec.append(['nems.xforms.split_at_time', {'valfrac': 0.1}])
#xfspec.append(['nems.xforms.split_by_occurrence_counts',
#               {'epoch_regex': '^STIM_'}])
#xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])

# MODEL SPEC
# modelspecname = 'dlog_wcg18x1_stp1_fir1x15_lvl1_dexp1'
modelspecname = 'wc.9x1.g-fir.1x15-lvl.1'
meta = {'cellid': cellid, 'batch': None, 'modelname': modelspecname}
xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])

xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_basic', {}])
xfspec.append(['nems.xforms.predict',    {}])
# xfspec.append(['nems.xforms.add_summary_statistics',    {}])
xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])
xfspec.append(['nems.xforms.plot_summary',    {}])

ctx = {}
ctx['stimfile'] = stimfile
ctx['respfile'] = respfile
ctx['exptid'] = exptid
ctx['cellid'] = "POL001-080"
ctx = xforms.evaluate_step(xfspec[0], ctx)

# actually do the fit
for xfa in xfspec[1:]:
    ctx = xforms.evaluate_step(xfa, ctx)


# Optional: uncomment to save your figure
# fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# browse the validation data
#aw = browse_recording(val[0], signals=['stim', 'pred', 'resp'], cellid=cellid)
aw = browse_context(ctx)


# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.

# TODO
