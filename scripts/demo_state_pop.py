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

datafile = os.path.join(signals_dir, "TAR010c.NAT.fs100.tgz")
exptid = "TAR010c"
batch = 307
cellid = "TAR010c-18-2"

# MODEL SPEC
modelspecname = 'stategain.SxN'

meta = {'cellid': cellid, 'batch': batch, 'modelname': modelspecname,
        'recording': exptid}
state_signals = ["pupil", "active", "population", "pupil_x_population", "active_x_population"]
jk_kwargs = {'njacks': 5}
xforms_init_context = {'cellid': cellid, 'batch': int(batch)}
xforms_init_context['keywordstring'] = modelspecname
xforms_init_context['meta'] = meta
xforms_init_context['recording_uri_list'] = [datafile]

# generate modelspec
xfspec = []
# load internally:
xfspec.append(['nems.xforms.init_context', xforms_init_context])
xfspec.append(['nems.xforms.load_recordings', {"save_other_cells_to_state": "population"}])
xfspec.append(['nems.xforms.make_state_signal',
              {'state_signals': state_signals, 'permute_signals': []}])
xfspec.append(["nems.xforms.mask_all_but_correct_references", {}])
xfspec.append(["nems.xforms.generate_psth_from_resp", {"smooth_resp": False, "use_as_input": True, "epoch_regex": "^STIM_"}])
xfspec.append(['nems.xforms.init_from_keywords', {}])
xfspec.append(['nems.xforms.mask_for_jackknife', jk_kwargs])
#xfspec.append(['nems.xforms.fit_state_init', {}])
xfspec.append(['nems.xforms.fit_basic', {}])

xfspec.append(['nems.xforms.predict', {}])
xfspec.append(['nems.xforms.add_summary_statistics', {}])
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
modelspec = ctx['modelspec']
modelspec.meta['cellid'] = cellid
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