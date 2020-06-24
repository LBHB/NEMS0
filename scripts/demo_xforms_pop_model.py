# A Template NEMS Script that demonstrates use of xforms for generating
# models that are easy to reload

import logging
import os

import nems.db as nd
import nems.uri
import nems.xforms as xforms

# ----------------------------------------------------------------------------
# CONFIGURATION

log = logging.getLogger(__name__)

# figure out data and results paths:
results_dir = nems.get_setting('NEMS_RESULTS_DIR')
signals_dir = nems.get_setting('NEMS_RECORDINGS_DIR')

# ----------------------------------------------------------------------------
# DATA LOADING & PRE-PROCESSING
#recording.get_demo_recordings(name="TAR010c_272b438ce3a5643e3e474206861096ce3ffdc000.tgz")

datafile = os.path.join(signals_dir, "TAR010c_afb264b3db970ec890e04c727e612c1cbfaced62.tgz")
datafile = os.path.join(signals_dir, "TAR010c.NAT.fs100.tgz")
datafile = os.path.join(signals_dir, "TAR010c_272b438ce3a5643e3e474206861096ce3ffdc000.tgz")
load_command = 'nems.demo.loaders.demo_loader'
exptid = "TAR010c"
batch = 271
siteid = "TAR010c"

# MODEL SPEC
#modelspecname = 'wc.18x1.g-fir.1x15-lvl.1'
modelspecname = 'dlog-wc.18x3.g-fir.1x10x3-relu.3-wc.3xR-lvl.R'
#modelspecname = 'dlog-wc.18x3.g-fir.1x10x3-relu.3-wc.3xR-lvl.R-dexp.R'

meta = {'siteid': siteid, 'batch': batch, 'modelname': modelspecname,
        'recording': exptid}

xforms_init_context = {'siteid': siteid, 'batch': int(batch)}
xforms_init_context['keywordstring'] = modelspecname
xforms_init_context['meta'] = meta
xforms_init_context['recording_uri_list'] = [datafile]

# generate modelspec
xfspec = []
# load internally:
xfspec.append(['nems.xforms.init_context', xforms_init_context])
xfspec.append(['nems.xforms.load_recordings', {}])
xfspec.append(['nems.preprocessing.resp_to_pc',
              {'pc_source': 'psth', 'overwrite_resp': False,
               'pc_count': 2}])
xfspec.append(['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': '^STIM_'}])
xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])

xfspec.append(['nems.xforms.init_from_keywords', {}])

#xfspec.append(['nems.xforms.fit_basic_init', {}])
#xfspec.append(['nems.xforms.fit_basic', {}])
#xfspec.append(['nems.analysis.fit_pop_model.init_pop_pca', {'flip_pcs': True}])
#xfspec.append(['nems.analysis.fit_pop_model.fit_population_iteratively',
#               {'fitter': 'scipy_minimize', 'tolerances': [1e-4, 3e-5],
#                'tol_iter': 50, 'fit_iter': 10}])
#xfspec.append(['nems.tf.cnnlink.fit_tf_init', {'early_stopping_tolerance': 5e-4}])
#xfspec.append(['nems.tf.cnnlink.fit_tf', {'early_stopping_tolerance': 1e-5}])
xfspec.append(['nems.tf.cnnlink_new.fit_tf_init', {'max_iter': 1000, 'early_stopping_tolerance': 5e-4}])
xfspec.append(['nems.tf.cnnlink_new.fit_tf', {'max_iter': 1000, 'early_stopping_tolerance': 1e-4}])
#xfspec.append(['nems.analysis.fit_pop_model.fit_population_iteratively',
#               {'fitter': 'scipy_minimize', 'tolerances': [1e-4, 3e-5],
#                'tol_iter': 50, 'fit_iter': 10}])


# xfspec.append(['nems.xforms.fit_basic_shrink', {}])
#xfspec.append(['nems.xforms.fit_basic_cd', {}])
# xfspec.append(['nems.xforms.fit_iteratively', {}])
xfspec.append(['nems.xforms.predict', {}])
xfspec.append(['nems.xforms.add_summary_statistics',    {}])
#xfspec.append(['nems.analysis.api.standard_correlation', {},
#               ['est', 'val', 'modelspec', 'rec'], ['modelspec']])

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
cellids=ctx['rec'].meta['cellid']
modelspec = ctx['modelspec']
modelspec.meta['cellid'] = siteid
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