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
recording_uri = signals_dir + "/TAR010c-18-1.tgz"
recordings = [recording_uri]
modelkeywords = 'dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1'
#modelkeywords = 'dlog-wc.18x2.g-stp.2-fir.2x15-dexp.1'
meta = {'cellids': ['TAR010c-18-1'], 'batch': 271, 'modelname': modelkeywords}

xfspec = [['load_recordings', {'recording_uri_list': recordings, 'meta': meta}],
          ['split_val_and_average_reps', {'epoch_regex': '^STIM_'}],
          ['init_from_keywords', {'keywordstring': modelkeywords}],
          ['fit_basic_init', {}],
          ['fit_basic', {}],
          ['predict', {}],
          ['add_summary_statistics', {}],
          ['plot_summary', {}]]

ctx, log_xf = xforms.evaluate(xfspec) # evaluate the fit script

#xforms.save_context(dest='/data/results', ctx=ctx, xfspec=xfspec, log=log_xf)

"""
Simplified:
import nems.recording as recording
import nems.xforms as xforms
    
recording.get_demo_recordings("/data/recordings/")
recordings = ["/data/recordings/TAR010c-18-1.tgz"]
modelkeywords = 'dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1'
#modelkeywords = 'dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1'
meta = {'cellid': 'TAR010c-18-1', 'batch': 271, 'modelname': modelkeywords}

xfspec = []
xfspec.append(['nems.xforms.load_recordings',
               {'recording_uri_list': recordings}])
xfspec.append(['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': '^STIM_'}])
xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])
xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelkeywords, 'meta': meta}])
xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_basic', {}])
xfspec.append(['nems.xforms.predict', {}])
xfspec.append(['nems.xforms.add_summary_statistics', {}])
xfspec.append(['nems.xforms.plot_summary', {}])

ctx, log_xf = xforms.evaluate(xfspec) # evaluate the fit script

#xforms.save_context(dest='/data/results/', ctx=ctx, xfspec=xfspec, log=log_xf)
"""

