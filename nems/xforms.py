import io
import os
import copy
import socket
import logging

import matplotlib.pyplot as plt
import numpy as np

import nems.analysis.api
import nems.initializers as init
import nems.metrics.api as metrics
import nems.modelspec as ms
from nems.modelspec import set_modelspec_metadata, get_modelspec_metadata,\
                           get_modelspec_shortname
import nems.plots.api as nplt
import nems.preprocessing as preproc
import nems.priors as priors
from nems.uri import save_resource, load_resource
from nems.utils import iso8601_datestring, find_module
from nems.fitters.api import scipy_minimize
from nems.recording import load_recording

log = logging.getLogger(__name__)
xforms = {}  # A mapping of kform keywords to xform 2-tuplets (2 element lists)


def defxf(keyword, xformspec):
    '''
    Adds xformspec to the xforms keyword dictionary.
    A helper function so not every keyword mapping has to be in a single
    file and part of a very large single multiline dict.
    '''
    if keyword in xforms:
        raise ValueError("Keyword already defined! Choose another name.")
    xforms[keyword] = xformspec


def load_xform(uri):
    '''
    Loads and returns xform saved as a JSON.
    '''
    xform = load_resource(uri)
    return xform


def xfspec_shortname(xformspec):
    '''
    Given an xformspec, makes a shortname for it.
    '''
    n = len('nems.xforms.')
    fn_names = [xf[n:] for xf, xfa in xformspec]
    name = ".".join(fn_names)
    return name


def evaluate_step(xfa, context={}):
    '''
    Helper function for evaluate. Take one step
    SVD revised 2018-03-23 so specialized xforms wrapper functions not required
      but now xfa can be len 4, where xfa[2] indicates context in keys and
      xfa[3] is context out keys
    '''
    
    if not(len(xfa) == 2 or len(xfa) == 4):
        raise ValueError('Got non 2- or 4-tuple for xform: {}'.format(xfa))
    xf = xfa[0]
    xfargs = xfa[1]
    if len(xfa) > 2:
        context_in = {k: context[k] for k in xfa[2]}
    else:
        context_in = context
    if len(xfa) > 3:
        context_out_keys = xfa[3]
    else:
        context_out_keys = []

    fn = ms._lookup_fn_at(xf)
    # Check for collisions; more to avoid confusion than for correctness:
    for k in xfargs:
        if k in context_in:
            m = 'xf arg {} overlaps with context: {}'.format(k, xf)
            raise ValueError(m)
    # Merge args into context, and make a deepcopy so that mutation
    # inside xforms will not be propogated unless the arg is returned.
    merged_args = {**xfargs, **context_in}
    args = copy.deepcopy(merged_args)
    # Run the xf
    log.info('Evaluating: {}'.format(xf))
    new_context = fn(**args)
    if len(context_out_keys):
        if type(new_context) is tuple:
            # print(new_context)
            new_context = {k: new_context[i] for i, k
                           in enumerate(context_out_keys)}
        elif len(context_out_keys) == 1:
            new_context = {context_out_keys[0]: new_context}
        else:
            raise ValueError('len(context_out_keys) needs to match '
                             'number of outputs from xf fun')
    # Use the new context for the next step
    if type(new_context) is not dict:
        raise ValueError('xf did not return a context dict: {}'.format(xf))
    context_out = {**context, **new_context}

    return context_out


def evaluate(xformspec, context={}, start=0, stop=None):
    '''
    Similar to modelspec.evaluate, but for xformspecs, which is a list of
    2-element lists of function and keyword arguments dict. Each XFORM must
    return a dictionary of the explicit changes made to the context, which
    is the dict that is passed from xform to xform.

    Also, this function wraps every logging call and saves it in a log
    that is the second value returned by this function.
    '''
    context = copy.deepcopy(context)  # Create a new starting context

    # Create a log stream set to the debug level; add it as a root log handler
    log_stream = io.StringIO()
    ch = logging.StreamHandler(log_stream)
    ch.setLevel(logging.DEBUG)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    ch.setFormatter(formatter)
    rootlogger = logging.getLogger()
    rootlogger.addHandler(ch)

    # Evaluate the xforms
    for xfa in xformspec[start:stop]:
        context = evaluate_step(xfa, context)

    # Close the log, remove the handler, and add the 'log' string to context
    log.info('Done (re-)evaluating xforms.')
    ch.close()
    rootlogger.removeFilter(ch)

    return context, log_stream.getvalue()


###############################################################################
# Stuff below this line are useful resuable components.
# See xforms_test.py for how to use it.


###############################################################################
##################       LOADERS / MODELSPEC STUFF    #########################
###############################################################################


def load_recordings(recording_uri_list, normalize=False, cellid=None,
                    save_other_cells_to_state=False, **context):
    '''
    Load one or more recordings into memory given a list of URIs.
    '''
    rec = load_recording(recording_uri_list[0])
    other_recordings = [load_recording(uri) for uri in recording_uri_list[1:]]
    if other_recordings:
        rec.concatenate_recordings(other_recordings)

    if normalize and 'stim' in rec.signals.keys():
        log.info('Normalizing stim')
        rec['stim'] = rec['stim'].rasterize().normalize('minmax')

    # if cellid is provided, use it to select channel or subset of channels
    # from resp signal.
    if cellid is None:
        pass

    elif type(cellid) is list:
        log.info('Extracting channels %s', cellid)
        excluded_cells = [cell for cell in rec['resp'].chans if cell in cellid]
        if save_other_cells_to_state is True:
            s = rec['resp'].extract_channels(excluded_cells).rasterize()

            rec = preproc.concatenate_state_channel(rec, s, 'state')
            rec['state_raw'] = rec['state']
        rec['resp'] = rec['resp'].extract_channels(cellid)

    elif cellid in rec['resp'].chans:
        log.info('Extracting channel %s', cellid)
        excluded_cells = rec['resp'].chans.copy()
        excluded_cells.remove(cellid)
        if save_other_cells_to_state is True:
            s = rec['resp'].extract_channels(excluded_cells).rasterize()
            rec = preproc.concatenate_state_channel(rec, s, 'state')
            rec['state_raw'] = rec['state']
        rec['resp'] = rec['resp'].extract_channels([cellid])

    else:
        log.info('No cellid match, keeping all resp channels')

    # Quick fix - will take care of this on the baphy loading side in the future.
    if 'pupil' in rec.signals.keys() and np.any(np.isnan(rec['pupil'].as_continuous())):
                log.info('Padding {0} with the last non-nan value'.format('pupil'))
                inds = ~np.isfinite(rec['pupil'].as_continuous())
                arr = copy.deepcopy(rec['pupil'].as_continuous())
                arr[inds] = arr[~inds][-1]
                rec['pupil'] = rec['pupil']._modified_copy(arr)

    return {'rec': rec}


def init_from_keywords(keywordstring, meta={}, IsReload=False,
                       registry=None, rec=None, **context):
    if not IsReload:
        modelspec = init.from_keywords(keyword_string=keywordstring,
                                       meta=meta, registry=registry, rec=rec)

        return {'modelspecs': [modelspec]}
    else:
        return {}


def load_modelspecs(modelspecs, uris,
                    IsReload=False, **context):
    '''
    i.e. Load a modelspec from a specific place. This is not
    the same as reloading a model for later inspection; it would be more
    appropriate when doing something complicated with several different
    models.
    '''
    if not IsReload:
        modelspecs = [load_resource(uri) for uri in uris]
    return {'modelspecs': modelspecs}


def set_random_phi(modelspecs, IsReload=False, **context):
    ''' Starts all modelspecs at random phi sampled from the priors. '''
    if not IsReload:
        modelspecs = [priors.set_random_phi(m) for m in modelspecs]
    return {'modelspecs': modelspecs}


def fill_in_default_metadata(rec, modelspecs, IsReload=False, **context):
    '''
    Sets any uninitialized metadata to defaults that should help us
    find it in nems_db again. (fitter, recording, date, etc)
    '''
    if not IsReload:
        # Add metadata to help you reload this state later
        for modelspec in modelspecs:
            meta = get_modelspec_metadata(modelspec)
            if 'fitter' not in meta:
                set_modelspec_metadata(modelspec, 'fitter', 'None')
            if 'fit_time' not in meta:
                set_modelspec_metadata(modelspec, 'fitter', 'None')
            if 'recording' not in meta:
                recname = rec.name if rec else 'None'
                set_modelspec_metadata(modelspec, 'recording', recname)
            if 'recording_uri' not in meta:
                uri = rec.uri if rec and rec.uri else 'None'
                set_modelspec_metadata(modelspec, 'recording_uri', uri)
            if 'date' not in meta:
                set_modelspec_metadata(modelspec, 'date', iso8601_datestring())
            if 'hostname' not in meta:
                set_modelspec_metadata(modelspec, 'hostname',
                                       socket.gethostname())
    return {'modelspecs': modelspecs}


def only_best_modelspec(modelspecs, metakey='r_test', comparison='greatest',
                        IsReload=False, **context):
    '''
    Collapses a list of modelspecs so that it only contains the modelspec
    with the highest given meta metric.
    '''
    if not IsReload:
        # TODO: Not the fastest way to do this but probably doesn't matter
        #       since it's only done once per fit.

        # TODO: Make this allow for variable number of top specs by
        #       updating ms function to sort then pick top n
        return {'modelspecs': ms.get_best_modelspec(modelspecs, metakey,
                                                    comparison)}
    else:
        return {}


def sort_modelspecs(modelspecs, metakey='r_test', order='descending',
                    IsReload=False, **context):
    '''
    Sorts modelspecs according to the specified metakey and order.
    '''
    if not IsReload:
        return {'modelspecs': ms.sort_modelspecs(modelspecs, metakey, order)}
    else:
        return {}


###############################################################################
#########################     PREPROCESSORS     ###############################
###############################################################################


def add_average_sig(rec, signal_to_average, new_signalname, epoch_regex,
                    **context):
    rec = preproc.add_average_sig(rec,
                                  signal_to_average=signal_to_average,
                                  new_signalname=new_signalname,
                                  epoch_regex=epoch_regex)
    return {'rec': rec}


def remove_all_but_correct_references(rec, **context):
    '''
    find REFERENCE epochs spanned by either PASSIVE_EXPERIMENT or
    HIT_TRIAL epochs. remove all other segments from signals in rec
    '''
    rec = preproc.remove_invalid_segments(rec)

    return {'rec': rec}


def mask_all_but_correct_references(rec, balance_rep_count=False,
                                    include_incorrect=False, **context):
    '''
    find REFERENCE epochs spanned by either PASSIVE_EXPERIMENT or
    HIT_TRIAL epochs. mask out all other segments from signals in rec
    '''
    rec = preproc.mask_all_but_correct_references(
            rec, balance_rep_count=balance_rep_count,
            include_incorrect=include_incorrect)

    return {'rec': rec}


def mask_all_but_targets(rec, **context):
    '''
    find TARGET epochs all behaviors/outcomes
    '''
    rec = preproc.mask_all_but_targets(rec)

    return {'rec': rec}


def generate_psth_from_resp(rec, epoch_regex='^STIM_',
                            smooth_resp=False, **context):
    '''
    generate PSTH prediction from rec['resp'] (before est/val split). Could
    be considered "cheating" b/c predicted PSTH then is based on data in
    val set, but this is because we're interested in testing state effects,
    not sensory coding models. The appropriate control, however is to run
    generate_psth_from_est_for_both_est_and_val_nfold on each nfold est/val
    split.
    '''

    rec = preproc.generate_psth_from_resp(rec, epoch_regex,
                                          smooth_resp=smooth_resp)

    return {'rec': rec}


def generate_psth_from_est_for_both_est_and_val_nfold(
        est, val, epoch_regex='^STIM_', **context):
    '''
    generate PSTH prediction for each set
    '''
    est_out, val_out = \
        preproc.generate_psth_from_est_for_both_est_and_val_nfold(est, val)
    return {'est': est_out, 'val': val_out}


def make_state_signal(rec, state_signals=['pupil'], permute_signals=[],
                      new_signalname='state', **context):

    rec = preproc.make_state_signal(rec, state_signals=state_signals,
                                    permute_signals=permute_signals,
                                    new_signalname=new_signalname)

    return {'rec': rec}


def make_mod_signal(rec, signal='resp'):
    """
    Make new signal called mod that can be used for calculating an unbiased
    mod_index
    """
    new_rec = rec.copy()
    psth = new_rec['psth_sp']
    resp = new_rec[signal]
    mod_data = resp.as_continuous() - psth.as_continuous()
    mod = psth._modified_copy(mod_data)
    mod.name = 'mod'
    new_rec.add_signal(mod)

    return new_rec


def split_by_occurrence_counts(rec, epoch_regex='^STIM_', **context):
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)

    return {'est': est, 'val': val}


def split_at_time(rec, valfrac=0.1, **context):

    rec['resp'] = rec['resp'].rasterize()
    rec['stim'] = rec['stim'].rasterize()
    est, val = rec.split_at_time(fraction=valfrac)

    return {'est': est, 'val': val}


def average_away_stim_occurrences(est, val, epoch_regex='^STIM_', **context):
    est = preproc.average_away_epoch_occurrences(est, epoch_regex=epoch_regex)
    val = preproc.average_away_epoch_occurrences(val, epoch_regex=epoch_regex)

    # mask out nan periods
#    d=np.isfinite(est['resp'].as_continuous()[[0],:])
#    log.info('found %d non-nans in est', np.sum(d))
#    est=est.create_mask()
#    est['mask']=est['mask']._modified_copy(d)
#
#    d=np.isfinite(val['resp'].as_continuous()[[0],:])
#    log.info('found %d non-nans  in val', np.sum(d))
#    val=val.create_mask()
#    val['mask']=val['mask']._modified_copy(d)

    return {'est': est, 'val': val}


def average_away_stim_occurrences_rec(rec, epoch_regex='^STIM_', **context):
    rec = preproc.average_away_epoch_occurrences(rec, epoch_regex=epoch_regex)
    return {'rec': rec}


def split_val_and_average_reps(rec, epoch_regex='^STIM_', **context):
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)
    est = preproc.average_away_epoch_occurrences(est, epoch_regex=epoch_regex)
    val = preproc.average_away_epoch_occurrences(val, epoch_regex=epoch_regex)

    return {'est': est, 'val': val}


def use_all_data_for_est_and_val(rec, **context):
    est = rec.copy()
    val = rec.copy()
    rec['resp'] = rec['resp'].rasterize()
    rec['stim'] = rec['stim'].rasterize()
    est['resp'] = est['resp'].rasterize()
    est['stim'] = est['stim'].rasterize()
    val['resp'] = val['resp'].rasterize()
    val['stim'] = val['stim'].rasterize()

    return {'rec': rec, 'est': est, 'val': val}


def split_for_jackknife(rec, modelspecs=None, epoch_name='REFERENCE',
                        njacks=10, IsReload=False, **context):

    est_out, val_out, modelspecs_out = \
        preproc.split_est_val_for_jackknife(rec, modelspecs=modelspecs,
                                            epoch_name=epoch_name,
                                            njacks=njacks, IsReload=IsReload)
    if IsReload:
        return {'est': est_out, 'val': val_out}
    else:
        return {'est': est_out, 'val': val_out, 'modelspecs': modelspecs_out}


def mask_for_jackknife(rec, modelspecs=None, epoch_name='REFERENCE',
                       by_time=False, njacks=10, IsReload=False, **context):

    if by_time != True:
        est_out, val_out, modelspecs_out = \
            preproc.mask_est_val_for_jackknife(rec, modelspecs=modelspecs,
                                               epoch_name=epoch_name,
                                               njacks=njacks, IsReload=IsReload)
    else:
        est_out, val_out, modelspecs_out = \
            preproc.mask_est_val_for_jackknife_by_time(rec, modelspecs=modelspecs,
                                               njacks=njacks, IsReload=IsReload)

    if IsReload:
        return {'est': est_out, 'val': val_out}
    else:
        return {'est': est_out, 'val': val_out, 'modelspecs': modelspecs_out}


def jack_subset(est, val, modelspecs=None, IsReload=False,
                keep_only=1, **context):

    if keep_only == 1:
        est = est[0]
        val = val[0]
        est['resp']=est['resp'].rasterize()
        val['resp']=val['resp'].rasterize()
        est['stim']=est['stim'].rasterize()
        val['stim']=val['stim'].rasterize()
    else:
        est = est[:keep_only]
        val = val[:keep_only]
    if modelspecs is not None:
        modelspecs_out = modelspecs[:keep_only]

    if IsReload:
        return {'est': est, 'val': val}
    else:
        return {'est': est, 'val': val, 'modelspecs': modelspecs_out}


###############################################################################
######################        INITIALIZERS         ############################
###############################################################################


def fit_basic_init(modelspecs, est, IsReload=False, metric='nmse',
                   tolerance=10**-5.5, norm_fir=False, nl_mode=2, **context):
    '''
    Initialize modelspecs in a way that avoids getting stuck in
    local minima.

    written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)

    '''
    # only run if fitting
    if not IsReload:
        if isinstance(metric, str):
            metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', 'resp')
        else:
            metric_fn = metric
        modelspecs = [nems.initializers.prefit_LN(
                est, modelspecs[0],
                analysis_function=nems.analysis.api.fit_basic,
                fitter=scipy_minimize, metric=metric_fn,
                tolerance=tolerance, max_iter=700, norm_fir=norm_fir, 
                nl_mode=nl_mode)]

    return {'modelspecs': modelspecs}


def _set_zero(x):
    y = x.copy()
    y[np.isfinite(y)] = 0
    return y


def fit_state_init(modelspecs, est, fit_sig='resp', tolerance=1e-4,
                   IsReload=False, metric='nmse', **context):
    '''
    Initialize modelspecs in an attempt to avoid getting stuck in
    local minima. Remove state replication/merging first.

    written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)

    assumption -- est['state'] signal is being used for merge
    '''
    if not IsReload:
        metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', 'resp')

        if type(est) is not list:
            # make est a list so that this function can handle standard
            # or n-fold fits
            est = [est]

        modelspecs_out = []
        i = 0
        for m, d in zip(modelspecs, est):
            i += 1
            log.info("Initializing modelspec %d/%d state-free",
                     i, len(modelspecs))

            # set state to 0 for all timepoints so that only first filterbank
            # is used
            dc = d.copy()
            dc['state'] = dc['state'].transform(_set_zero, 'state')
            if fit_sig != 'resp':
                log.info("Subbing %s for resp signal", fit_sig)
                dc['resp'] = dc[fit_sig]

            m = nems.initializers.prefit_LN(
                    dc, m,
                    analysis_function=nems.analysis.api.fit_basic,
                    fitter=scipy_minimize, metric=metric_fn,
                    tolerance=tolerance, max_iter=700)
            # fit a bit more to settle in STP variables and anything else
            # that might have been excluded
            fit_kwargs = {'tolerance': tolerance/2, 'max_iter': 500}
            m = nems.analysis.api.fit_basic(
                    dc, m, fit_kwargs=fit_kwargs, metric=metric_fn,
                    fitter=scipy_minimize)[0]
            rep_idx = find_module('replicate_channels', m)
            mrg_idx = find_module('merge_channels', m)
            if rep_idx is not None:
                repcount = m[rep_idx]['fn_kwargs']['repcount']
                for j in range(rep_idx+1, mrg_idx):
                    # assume all phi
                    log.debug(m[j]['fn'])
                    if 'phi' in m[j].keys():
                        for phi in m[j]['phi'].keys():
                            s = m[j]['phi'][phi].shape
                            setcount = int(s[0] / repcount)
                            log.debug('phi[%s] setcount=%d', phi, setcount)
                            snew = np.ones(len(s))
                            snew[0] = repcount
                            new_v = np.tile(m[j]['phi'][phi][:setcount, ...],
                                            snew.astype(int))
                            log.debug(new_v)
                            m[j]['phi'][phi] = new_v

            modelspecs_out.append(m)

        modelspecs = modelspecs_out

    return {'modelspecs': modelspecs}


###############################################################################
########################       FITTERS / ANALYSES      ########################
###############################################################################


def fit_basic(modelspecs, est, max_iter=1000, tolerance=1e-7,
              metric='nmse', IsReload=False, fitter='scipy_minimize',
              jackknifed_fit=False, random_sample_fit=False,
              n_random_samples=0, random_fit_subset=None, **context):
    ''' A basic fit that optimizes every input modelspec. '''
    if not IsReload:
        metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', 'resp')
        fitter_fn = getattr(nems.fitters.api, fitter)
        fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

        if jackknifed_fit:
            return fit_nfold(modelspecs, est, tolerance=tolerance,
                             metric=metric, fitter=fitter,
                             fit_kwargs=fit_kwargs, analysis='fit_basic',
                             **context)

        elif random_sample_fit:
            basic_kwargs = {'metric': metric_fn, 'fitter': fitter_fn,
                            'fit_kwargs': fit_kwargs}
            return fit_n_times_from_random_starts(
                        modelspecs, est, ntimes=n_random_samples,
                        subset=random_fit_subset,
                        analysis='fit_basic', basic_kwargs=basic_kwargs
                        )

        else:
            # standard single shot
            modelspecs = [
                    nems.analysis.api.fit_basic(est, modelspec,
                                                fit_kwargs=fit_kwargs,
                                                metric=metric_fn,
                                                fitter=fitter_fn)[0]
                    for modelspec in modelspecs
                    ]

    return {'modelspecs': modelspecs}


def fit_iteratively(modelspecs, est, tol_iter=100, fit_iter=20, IsReload=False,
                    module_sets=None, invert=False, tolerances=[1e-4],
                    metric='nmse', fitter='scipy_minimize', fit_kwargs={},
                    jackknifed_fit=False, random_sample_fit=False,
                    n_random_samples=0, random_fit_subset=None, **context):

    fitter_fn = getattr(nems.fitters.api, fitter)
    metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', 'resp')

    if not IsReload:
        if jackknifed_fit:
            return fit_nfold(modelspecs, est, tol_iter=tol_iter,
                             fit_iter=fit_iter, module_sets=module_sets,
                             tolerances=tolerances, metric=metric,
                             fitter=fitter, fit_kwargs=fit_kwargs,
                             analysis='fit_iteratively', **context)

        elif random_sample_fit:
            iter_kwargs = {'tol_iter': tol_iter, 'fit_iter': fit_iter,
                           'invert': invert, 'tolerances': tolerances,
                           'module_sets': module_sets, 'metric': metric_fn,
                           'fitter': fitter_fn, 'fit_kwargs': fit_kwargs}
            return fit_n_times_from_random_starts(
                        modelspecs, est, ntimes=n_random_samples,
                        subset=random_fit_subset,
                        analysis='fit_iteratively', iter_kwargs=iter_kwargs,
                        )

        else:
            modelspecs = [
                    nems.analysis.api.fit_iteratively(
                            est, modelspec, fit_kwargs=fit_kwargs,
                            fitter=fitter_fn, module_sets=module_sets,
                            invert=invert, tolerances=tolerances,
                            tol_iter=tol_iter, fit_iter=fit_iter,
                            metric=metric_fn)[0]
                    for modelspec in modelspecs
                    ]

    return {'modelspecs': modelspecs}


def fit_nfold(modelspecs, est, tolerance=1e-7, max_iter=1000,
              IsReload=False, metric='nmse', fitter='scipy_minimize',
              analysis='fit_basic', tolerances=None, module_sets=None,
              tol_iter=100, fit_iter=20, **context):
    ''' fitting n fold, one from each entry in est '''
    if not IsReload:
        metric = lambda d: getattr(metrics, metric)(d, 'pred', 'resp')
        fitter_fn = getattr(nems.fitters.api, fitter)
        fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}
        if fitter == 'coordinate_descent':
            fit_kwargs['step_size'] = 0.1
        modelspecs = nems.analysis.api.fit_nfold(
                est, modelspecs, fitter=fitter_fn,
                fit_kwargs=fit_kwargs, analysis=analysis,
                tolerances=tolerances, module_sets=module_sets,
                tol_iter=tol_iter, fit_iter=fit_iter)

    return {'modelspecs': modelspecs}


def fit_n_times_from_random_starts(modelspecs, est, ntimes, subset,
                                   analysis='fit_basic', basic_kwargs={},
                                   IsReload=False, **context):
    ''' Self explanatory. '''
    if not IsReload:
        if len(modelspecs) > 1:
            raise NotImplementedError('I only work on 1 modelspec')

        modelspecs = nems.analysis.api.fit_from_priors(
                est, modelspecs[0], ntimes=ntimes, subset=subset,
                analysis=analysis, basic_kwargs=basic_kwargs
                )

    return {'modelspecs': modelspecs}


###############################################################################
########################         SAVE / SUMMARY        ########################
###############################################################################


def save_recordings(modelspecs, est, val, **context):
    # TODO: Save the recordings somehow?
    return {'modelspecs': modelspecs}


def predict(modelspecs, est, val, **context):
    # modelspecs = metrics.add_summary_statistics(est, val, modelspecs)
    # TODO: Add statistics to metadata of every modelspec

    est, val = nems.analysis.api.generate_prediction(est, val, modelspecs)

    return {'val': val, 'est': est}


def add_summary_statistics(est, val, modelspecs, fn='standard_correlation',
                           rec=None, use_mask=True, **context):
    '''
    standard_correlation: average all correlation metrics and add
                          to first modelspec only.
    correlation_per_model: evaluate correlation metrics separately for each
                           modelspec and save results in each modelspec
    '''
    corr_fn = getattr(nems.analysis.api, fn)
    modelspecs = corr_fn(est, val, modelspecs, rec=rec, use_mask=use_mask)

    if find_module('state', modelspecs[0]) is not None:
        s = metrics.state_mod_index(val[0], epoch='REFERENCE', psth_name='pred',
                            state_sig='state_raw', state_chan=[])
        j_s, ee = metrics.j_state_mod_index(val[0], epoch='REFERENCE', psth_name='pred',
                            state_sig='state_raw', state_chan=[], njacks=10)
        modelspecs[0][0]['meta']['state_mod'] = s
        modelspecs[0][0]['meta']['j_state_mod'] = j_s
        modelspecs[0][0]['meta']['se_state_mod'] = ee
        modelspecs[0][0]['meta']['state_chans'] = val[0]['state'].chans

        # Charlie testing diff ways to calculate mod index

        # try using resp
        s = metrics.state_mod_index(val[0], epoch='REFERENCE', psth_name='resp',
                            state_sig='state_raw', state_chan=[])
        j_s, ee = metrics.j_state_mod_index(val[0], epoch='REFERENCE', psth_name='resp',
                            state_sig='state_raw', state_chan=[], njacks=10)
        modelspecs[0][0]['meta']['state_mod_r'] = s
        modelspecs[0][0]['meta']['j_state_mod_r'] = j_s
        modelspecs[0][0]['meta']['se_state_mod_r'] = ee

        # try using the "mod" signal (if it exists) which is calculated
        if 'mod' in modelspecs[0][0]['meta']['modelname']:
            s = metrics.state_mod_index(val[0], epoch='REFERENCE',
                                            psth_name='mod', divisor='resp',
                                            state_sig='state_raw', state_chan=[])
            j_s, ee = metrics.j_state_mod_index(val[0], epoch='REFERENCE',
                                            psth_name='mod', divisor='resp',
                                            state_sig='state_raw', state_chan=[],
                                            njacks=10)
            modelspecs[0][0]['meta']['state_mod_m'] = s
            modelspecs[0][0]['meta']['j_state_mod_m'] = j_s
            modelspecs[0][0]['meta']['se_state_mod_m'] = ee

    return {'modelspecs': modelspecs}


def plot_summary(modelspecs, val, figures=None, IsReload=False, **context):
    # CANNOT initialize figures=[] in optional args our you will create a bug

    if figures is None:
        figures = []
    if not IsReload:
        fig = nplt.quickplot({'modelspecs': modelspecs, 'val': val})
        # Needed to make into a Bytes because you can't deepcopy figures!
        figures.append(nplt.fig2BytesIO(fig))

    return {'figures': figures}


###############################################################################
########################            FLAGS              ########################
###############################################################################


def use_metric(metric='nmse_shrink', IsReload=False, **context):
    if not IsReload:
        return {'metric': metric}
    else:
        return {}


def jackknifed_fit(IsReload=False, **context):
    if not IsReload:
        return {'jackknifed_fit': True}
    else:
        return {}


def random_sample_fit(ntimes=10, subset=None, IsReload=False, **context):
    if not IsReload:
        return {'random_sample_fit': True, 'n_random_samples': ntimes,
                'random_fit_subset': subset}
    else:
        return {}


# TODO: Perturb around the modelspec to get confidence intervals

# TODO: Use simulated annealing (Slow, arguably gets stuck less often)
# modelspecs = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.annealing)

# TODO: Use Metropolis algorithm (Very slow, gives confidence interval)
# modelspecs = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.metropolis)

# TODO: Use 10-fold cross-validated evaluation
# fitter = partial(nems.cross_validator.cross_validate_wrapper, gradient_descent, 10)
# modelspecs = nems.analysis.fit_cv(est, modelspec, folds=10)


###############################################################################
##################        XFORMS UTILITIES             ########################
###############################################################################


def tree_path(recording, modelspecs, xfspec):
    '''
    Returns a relative path (excluding filename, host, port) for URIs.
    Editing this function edits the path in the file tree of every
    file saved!
    '''

    xformname = xfspec_shortname(xfspec)
    modelname = get_modelspec_shortname(modelspecs[0])
    recname = recording.name  # Or from rec.uri???
    meta = get_modelspec_metadata(modelspecs[0])
    date = meta.get('date', iso8601_datestring())

    path = '/' + recname + '/' + modelname + '/' + xformname + '/' + date + '/'

    return path


def save_analysis(destination,
                  recording,
                  modelspecs,
                  xfspec,
                  figures,
                  log,
                  add_tree_path=False):
    '''Save an analysis file collection to a particular destination.'''
    if add_tree_path:
        treepath = tree_path(recording, modelspecs, xfspec)
        base_uri = os.path.join(destination, treepath)
    else:
        base_uri = destination

    base_uri = base_uri if base_uri[-1] == '/' else base_uri + '/'
    xfspec_uri = base_uri + 'xfspec.json'  # For attaching to modelspecs

    for number, modelspec in enumerate(modelspecs):
        set_modelspec_metadata(modelspec, 'xfspec', xfspec_uri)
        save_resource(base_uri + 'modelspec.{:04d}.json'.format(number),
                      json=modelspec)
    for number, figure in enumerate(figures):
        save_resource(base_uri + 'figure.{:04d}.png'.format(number),
                      data=figure)
    save_resource(base_uri + 'log.txt', data=log)
    save_resource(xfspec_uri, json=xfspec)
    return {'savepath': base_uri}


def load_analysis(filepath, eval_model=True, only=None):
    """
    load xforms and modelspec(s) from a specified directory
    """
    log.info('Loading modelspecs from %s...', filepath)

    xfspec = load_xform(filepath + 'xfspec.json')

    mspaths = []
    for file in os.listdir(filepath):
        if file.startswith("modelspec"):
            mspaths.append(filepath + "/" + file)
    ctx = load_modelspecs([], uris=mspaths, IsReload=False)
    ctx['IsReload'] = True

    if eval_model:
        ctx, log_xf = evaluate(xfspec, ctx)
    elif only is not None:
        # Useful for just loading the recording without doing
        # any subsequent evaluation.
        ctx, log_xf = evaluate([xfspec[only]], ctx)

    return xfspec, ctx


###############################################################################
##################        CONTEXT UTILITIES             #######################
###############################################################################


def evaluate_context(ctx, rec_key='val', rec_idx=0, mspec_idx=0, start=None,
                     stop=None):
    rec = ctx[rec_key][0]
    mspec = ctx['modelspecs'][mspec_idx]
    return ms.evaluate(rec, mspec, start=start, stop=stop)


def get_meta(ctx, mspec_idx=0, mod_idx=0):
    return ctx['modelspecs'][mspec_idx][mod_idx]['meta']


def get_modelspec(ctx, mspec_idx=0):
    return ctx['modelspecs'][mspec_idx]


def get_module(ctx, val, key='index', mspec_idx=0, find_all_matches=False):
    mspec = ctx['modelspecs'][mspec_idx]
    if key in ['index', 'idx', 'i']:
        return mspec[val]
    else:
        i = find_module(val, mspec, find_all_matches=find_all_matches, key=key)
        return mspec[i]


def plot_heatmap(ctx, signal_name, cutoff=None, rec_key='val', rec_idx=0,
                 mspec_idx=0, start=None, stop=None):

    array = get_signal_as_array(ctx, signal_name, cutoff, rec_key, rec_idx,
                                mspec_idx, start, stop)
    array = array[:, ~np.all(np.isnan(array), axis=0)]
    plt.imshow(array, aspect='auto')


def plot_timeseries(ctx, signal_name, cutoff=None, rec_key='val', rec_idx=0,
                    mspec_idx=0, start=None, stop=None):

    array = get_signal_as_array(ctx, signal_name, cutoff, rec_key, rec_idx,
                                mspec_idx, start, stop)
    plt.plot(array.T)


def get_signal_as_array(ctx, signal_name, cutoff=None, rec_key='val',
                        rec_idx=0, mspec_idx=0, start=None, stop=None):

    rec = evaluate_context(ctx, rec_key=rec_key, rec_idx=rec_idx,
                           mspec_idx=mspec_idx, start=start, stop=stop)

    array = rec[signal_name].as_continuous()
    if cutoff is not None:
        if isinstance(cutoff, int):
            array = array[:, :cutoff]
        elif isinstance(cutoff, tuple):
            array = array[:, cutoff[0]:cutoff[1]]
        else:
            raise ValueError("cutoff must be an integer (:cutoff)"
                             "or a tuple (cutoff[0]:cutoff[1])")

    return array


###############################################################################
########################          UNUSED?        ##############################
###############################################################################


def fit_random_subsets(modelspecs, est, nsplits,
                       IsReload=False, **context):
    ''' Randomly sample parts of the data? Wait, HOW DOES THIS WORK? TODO?'''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = nems.analysis.api.fit_random_subsets(est,
                                                          modelspecs[0],
                                                          nsplits=nsplits)
    return {'modelspecs': modelspecs}


def fit_equal_subsets(modelspecs, est, nsplits,
                      IsReload=False, **context):
    ''' Divide the data into nsplits equal pieces and fit each one.'''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = nems.analysis.api.fit_subsets(est,
                                                   modelspecs[0],
                                                   nsplits=nsplits)
    return {'modelspecs': modelspecs}


def fit_jackknifes(modelspecs, est, njacks,
                   IsReload=False, **context):
    ''' Jackknife the data, fit on those, and make predictions from those.'''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = nems.analysis.api.fit_jackknifes(est,
                                                      modelspecs[0],
                                                      njacks=njacks)
    return {'modelspecs': modelspecs}


def fit_module_sets(modelspecs, est, max_iter=1000, IsReload=False,
                    module_sets=None, invert=False, tolerance=1e-4,
                    fitter=scipy_minimize, fit_kwargs={}, **context):

    if not IsReload:
        if len(modelspecs) > 1:
            raise NotImplementedError("Not supported for multiple modelspecs")
        modelspecs = [
                nems.analysis.api.fit_module_sets(
                        est, modelspec, fit_kwargs=fit_kwargs,
                        fitter=fitter, module_sets=module_sets,
                        invert=invert, tolerance=tolerance,
                        max_iter=max_iter)[0]
                for modelspec in modelspecs
                ]
    return {'modelspecs': modelspecs}
