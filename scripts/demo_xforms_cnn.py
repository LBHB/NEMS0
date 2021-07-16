# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import logging
import os

from nems import recording
import nems.db as nd
import nems.uri
import nems.xforms as xforms

# ----------------------------------------------------------------------------
# GLOBAL CONFIGURATION

log = logging.getLogger(__name__)

# figure out data and results paths:
results_dir = nems.get_setting('NEMS_RESULTS_DIR')
signals_dir = nems.get_setting('NEMS_RECORDINGS_DIR')

# ----------------------------------------------------------------------------
# DATA LOADING & PRE-PROCESSING

# Download demo recordings
recording.get_demo_recordings()

datafile = os.path.join(signals_dir, "TAR010c.NAT.fs100.ch18.tgz")
exptid = "TAR010c"
batch = 271
cellid = "TAR010c-18-2"

# SINGLE MODEL SPEC

# LN model
modelspecname = 'dlog-wc.18x3.g-fir.3x15-relu.1'

# Simple CNN
#modelspecname = 'wc.18x3-fir.1x10x3-relu.3-wc.3xR-lvl.R'


meta = {'cellid': cellid, 'batch': batch, 'modelname': modelspecname,
        'recording': exptid}

# cellid indicates a single cell
xforms_init_context = {'cellid': cellid, 'batch': int(batch)}
xforms_init_context['keywordstring'] = modelspecname
xforms_init_context['meta'] = meta
xforms_init_context['recording_uri_list'] = [datafile]

# generate sequence of xforms commands to load, initialize, fit and evaluate the model
xfspec = []

# load
xfspec.append(['nems.xforms.init_context', xforms_init_context])
xfspec.append(['nems.xforms.load_recordings', {}])

# estimation / validation split
xfspec.append(['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': '^STIM_'}])
xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])

# initialize modelspec
xfspec.append(['nems.xforms.init_from_keywords', {}])

# fit coarsely without final nonlinearity
xfspec.append(['nems.tf.cnnlink_new.fit_tf_init', {'max_iter': 1000, 'early_stopping_tolerance': 5e-4}])

# fit with final NL
xfspec.append(['nems.tf.cnnlink_new.fit_tf', {'max_iter': 1000, 'early_stopping_tolerance': 1e-4}])

# predict response in validation data
xfspec.append(['nems.xforms.predict', {}])

# measure prediction accuracy
xfspec.append(['nems.xforms.add_summary_statistics',    {}])

# GENERATE PLOTS
xfspec.append(['nems.xforms.plot_summary', {}])

# actually do the fit
log_xf = "NO LOG"
ctx = {}

# run each xforms command
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)
#ctx, log_xf = xforms.evaluate(xfspec)

# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# save results to file
cellids=ctx['rec'].meta['cellid']
modelspec = ctx['modelspec']
modelspec.meta['cellid'] = cellid

# save location generated when model initialized. Can be changed
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
