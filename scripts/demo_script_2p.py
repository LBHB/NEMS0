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
import nems.plots.api as nplt
import nems.modelspec as ms
import nems.xforms as xforms
import nems.utils
import nems.uri
import nems.recording as recording
from nems.signal import RasterizedSignal
from nems.fitters.api import scipy_minimize
import nems.db as nd

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# LOAD AND FORMAT RECORDING DATA

# data file and results locations
# defined in nems/nems/configs/settings.py, which will override
# defaults in nems/nems/configs/defaults.py
results_dir = nems.get_setting('NEMS_RESULTS_DIR')
recordings_dir = nems.get_setting('NEMS_RECORDINGS_DIR')

save_results = True
browse_results = True

# 2p data from Polley Lab at EPL
respfile = os.path.join(recordings_dir, 'data_nems_2p/neurons.csv')
stimfile = os.path.join(recordings_dir, 'data_nems_2p/stim_spectrogram.csv')
exptid = "POL001"
cellid = "POL001-080"
batch = 1  # define the group of data this belong to (eg, 1: A1, 2: AAF, etc)
load_command='nems.demo.loaders.load_polley_data'

# MODEL SPEC
# modelspecname = 'dlog_wcg18x1_stp1_fir1x15_lvl1_dexp1'
modelspecname = 'wc.9x1.g-fir.1x15-lvl.1'
meta = {'cellid': cellid, 'batch': batch, 'modelname': modelspecname}

xfspec = []
xfspec.append(['nems.xforms.load_recording_wrapper',
               {'load_command': load_command,
                'respfile': respfile,
                'stimfile': stimfile,
                'exptid': exptid,
                'cellid': cellid}])

# reserve 10% of the data for validation
xfspec.append(['nems.xforms.split_at_time', {'valfrac': 0.1}])

xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])

xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_basic', {}])
xfspec.append(['nems.xforms.predict',    {}])
# xfspec.append(['nems.xforms.add_summary_statistics',    {}])
xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspec', 'rec'], ['modelspec']])
xfspec.append(['nems.xforms.plot_summary',    {}])

# actually do the fit
ctx, log_xf = xforms.evaluate(xfspec)

# shorthand for:
# ctx = {}
# for xfa in xfspec:
#     ctx = xforms.evaluate_step(xfa, ctx)

if browse_results:
    import nems.gui.editors as gui
    ex = gui.browse_xform_fit(ctx, xfspec)

if save_results:
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
    modelspec.meta['modelpath'] = destination
    modelspec.meta['figurefile'] = destination + 'figure.0000.png'
    nd.update_results_table(modelspec)

# reload using:
"""
# repeated from above
import nems.db as nd
import nems.xforms as xforms

# pick your cell/batch/model
cellid = "POL001-080"
batch = 1  # define the group of data this belong to (eg, 1: A1, 2: AAF, etc)
modelspecname = 'wc.9x1.g-fir.1x15-lvl.1'

# find the results in the database
d=nd.get_results_file(batch=batch, cellids=[cellid], modelnames=[modelspecname])
d.loc[0]
filepath = d['modelpath'][0] + '/'

# load and display results
xfspec, ctx = xforms.load_analysis(filepath, eval_model=True)
"""

# ----------------------------------------------------------------------------
# BROWSE YOUR VALIDATION DATA
#aw = browse_context(ctx)

