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
nems_dir = os.path.abspath(os.path.dirname(xforms.__file__) + '/..')
signals_dir = nems_dir + '/recordings'
modelspecs_dir = nems_dir + '/modelspecs'

# ----------------------------------------------------------------------------
# DATA LOADING & PRE-PROCESSING
#recording_uri = signals_dir + "/por074b-c2.tgz"
#recordings = [recording_uri]
#cellid='por074b-c2'
#batch=259
recording_uri = signals_dir + "/TAR010c-18-1.tgz"
recordings = [recording_uri]
cellid='TAR010c-18-1'
batch=271

xfspec = []
xfspec.append(['nems.xforms.load_recordings',
               {'recording_uri_list': recordings}])
xfspec.append(['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': '^STIM_'}])
xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])

# MODEL SPEC
# modelspecname = 'dlog_wcg18x1_stp1_fir1x15_lvl1_dexp1'
#modelspecname = 'dlog_fir2x15_lvl1_dexp1'
modelspecname = 'dlog_wcg18x1_fir1x15_lvl1_dexp1'

meta = {'cellid': cellid, 'batch': batch, 'modelname': modelspecname}

xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])

xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_basic', {'maxiter': 1000, 'ftol': 1e-5}])
# xfspec.append(['nems.xforms.fit_basic', {}])
# xfspec.append(['nems.xforms.fit_iteratively', {}])

xfspec.append(['nems.xforms.predict',    {}])
xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])
# xfspec.append(['nems.xforms.add_summary_statistics',    {}])

# GENERATE PLOTS
xfspec.append(['nems.xforms.plot_summary',    {}])

# actually do the fit
ctx, log_xf = xforms.evaluate(xfspec)



# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# GOAL: Save your results to disk. (BEFORE you screw it up trying to plot!)

# logging.info('Saving Results...')
# ms.save_modelspecs(modelspecs_dir, modelspecs)

