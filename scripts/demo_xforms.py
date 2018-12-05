# A Template NEMS Script that demonstrates use of xforms for generating
# models that are easy to reload

import os
import logging
import sys
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
import nems.db as nd

from nems.recording import Recording
from nems.fitters.api import scipy_minimize

# ----------------------------------------------------------------------------
# CONFIGURATION

log = logging.getLogger(__name__)

# figure out data and results paths:
results_dir = nems.get_setting('NEMS_RESULTS_DIR')
signals_dir = nems.get_setting('NEMS_RECORDINGS_DIR')

# ----------------------------------------------------------------------------
# DATA LOADING & PRE-PROCESSING
recording_uri = os.path.join(signals_dir, "TAR010c-18-1.tgz")

datafile = os.path.join(signals_dir, "TAR010c-18-1.pkl")
load_command = 'nems.demo.loaders.demo_loader'
exptid = "TAR010c"
batch = 271
cellid = "TAR010c-18-1"

# MODEL SPEC
#modelspecname = 'dlog-wc.18x1.g-stp.1-fir.1x15-lvl.1-dexp.1'
modelspecname = 'wc.18x1.g-fir.1x15-lvl.1'


# generate modelspec
xfspec = []
# load internally:
#xfspec.append(['nems.xforms.load_recordings',
#               {'recording_uri_list': [recording_uri]}])
# load from external format
xfspec.append(['nems.xforms.load_recording_wrapper',
               {'load_command': load_command,
                'exptid': exptid,
                'datafile': datafile}])
xfspec.append(['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': '^STIM_'}])
xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])

# MODEL SPEC
#modelspecname = 'dlog-wc.18x1.g-stp.1-fir.1x15-lvl.1-dexp.1'
#modelspecname = 'wc.18x1.g-fir.1x15-lvl.1'
modelspecname = 'dlog-wc.18x1.g-fir.1x15-lvl.1'

meta = {'cellid': cellid, 'batch': batch, 'modelname': modelspecname,
        'recording': exptid}

xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])

xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_basic', {}])
# xfspec.append(['nems.xforms.fit_basic_shrink', {}])
#xfspec.append(['nems.xforms.fit_basic_cd', {}])
# xfspec.append(['nems.xforms.fit_iteratively', {}])
xfspec.append(['nems.xforms.predict', {}])
# xfspec.append(['nems.xforms.add_summary_statistics',    {}])
xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspec', 'rec'], ['modelspec']])

# GENERATE PLOTS
xfspec.append(['nems.xforms.plot_summary', {}])

# actually do the fit
log_xf = "NO LOG"
ctx = {}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)
#ctx, log_xf = xforms.evaluate(xfspec)


# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# save results to file
destination = os.path.join(results_dir, str(batch), xforms.get_meta(ctx)['cellid'],
                           ms.get_modelspec_longname(ctx['modelspec']))
log.info('Saving modelspec(s) to {0} ...'.format(destination))
xforms.save_analysis(destination,
                      recording=ctx['rec'],
                      modelspec=ctx['modelspec'],
                      xfspec=xfspec,
                      figures=ctx['figures'],
                      log=log_xf)

# save summary of results to a database
log.info('Saving metadata to db  ...')
modelspec = ctx['modelspec']
modelspec.meta()['modelpath'] = destination
modelspec.meta()['figurefile'] = destination + 'figure.0000.png'
nd.update_results_table(modelspec)
