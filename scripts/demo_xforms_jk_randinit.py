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
import nems.recording as recording
from nems.fitters.api import scipy_minimize, coordinate_descent
from nems.gui.editors import browse_xform_fit

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

# exptid = "sti023c"
# cellid = "sti023c-b1"
# datafile = os.path.join(signals_dir, "274", "sti023c_77852fc2f56f6be6be644ce9bd1377f301f8ccc1.tgz")
exptid = "eno006b"
cellid = "eno006b-a1"
datafile = os.path.join(signals_dir, "274", "eno006b_df3a28e2983290acb1ddbc47e0f58212abff2ade.tgz")
load_command = 'nems.demo.loaders.demo_loader'
batch = 274

# MODEL SPEC
modelspecname = 'dlog-do.2x10-relu.1'
modelspecname = 'dlog.f-wc.2x1.c-do.1x15-lvl.1-dexp.1'

meta = {'cellid': cellid, 'batch': batch, 'modelname': modelspecname,
        'recording': exptid}

xforms_init_context = {'cellid': cellid, 'batch': int(batch)}
xforms_init_context['keywordstring'] = modelspecname
xforms_init_context['meta'] = meta
xforms_init_context['recording_uri_list'] = [datafile]

# generate modelspec
xfspec = []
# load internally:
xfspec.append(['nems.xforms.init_context', xforms_init_context])
xfspec.append(['nems.xforms.load_recordings', {}])

xfspec.append(['nems.xforms.make_state_signal',
 {'state_signals': ['active'], 'permute_signals': [],
  'new_signalname': 'state'}])
xfspec.append(['nems.xforms.mask_all_but_correct_references',
 {'balance_rep_count': False, 'include_incorrect': False,
  'generate_evoked_mask': False}])

xfspec.append(['nems.xforms.init_from_keywords', {}])

xfspec.append(['nems.xforms.mask_for_jackknife', {'njacks': 3}])

# shortcut for testing with jackknife off
#xfspec.append(['nems.xforms.jack_subset', {'keep_only': 1}])

xfspec.append(['nems.initializers.rand_phi', {'rand_count': 4}])
xfspec.append(['nems.xforms.fit_state_init', {}])
xfspec.append(['nems.xforms.fit_basic', {}])

xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

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

""" """
# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# save results to file
cellids=ctx['rec'].meta['cellid']
modelspec = ctx['modelspec']
modelspec.meta['cellids'] = cellids
#destination = os.path.join(results_dir, str(batch), modelspec.meta['cellid'],
#                           modelspec.get_longname())
#modelspec.meta['modelpath'] = destination
#modelspec.meta['figurefile'] = destination + 'figure.0000.png'
log.info('Saving modelspec(s) to {0} ...'.format(modelspec.meta['modelpath']))
xforms.save_analysis(modelspec.meta['modelpath'],
                     recording=ctx['rec'],
                     modelspec=modelspec,
                     xfspec=xfspec,
                     figures=ctx['figures'],
                     log=log_xf)

# save summary of results to a database
log.info('Saving metadata to db  ...')
nd.update_results_table(modelspec)

# browse_xform_fit(ctx, xfspec, recname='val')

""" """