import io
import os
import copy
import socket
import logging

import matplotlib.pyplot as plt
import numpy as np

import nems.db as nd
import nems.analysis.api
import nems.initializers as init
import nems.metrics.api as metrics
import nems.modelspec as ms
from nems.modelspec import set_modelspec_metadata, get_modelspec_metadata,\
                           get_modelspec_shortname
import nems.plots.api as nplt
import nems.preprocessing as preproc
import nems.priors as priors
from nems import get_setting
from nems.registry import KeywordRegistry
from nems.plugins import (default_keywords, default_loaders, default_fitters,
                          default_initializers)
from nems.signal import RasterizedSignal
from nems.uri import save_resource, load_resource
from nems.utils import (iso8601_datestring, find_module,
                        recording_filename_hash, get_default_savepath)
from nems.fitters.api import scipy_minimize
from nems.recording import load_recording, Recording

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
        # backward compatibility
        a = [k.replace("modelspecs", "modelspec") for k in xfa[2]]
        context_in = {k: context[k] for k in a}
    else:
        context_in = context
    if len(xfa) > 3:
        a = [k.replace("modelspecs", "modelspec") for k in xfa[3]]
        context_out_keys = a
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
    logstring = log_stream.getvalue()
    context['log'] = logstring

    return context, logstring


###############################################################################
# Stuff below this line are useful resuable components.
# See xforms_test.py for how to use it.


###############################################################################
##################       LOADERS / MODELSPEC STUFF    #########################
###############################################################################


def init_context(kw_kwargs=None, **context):
    #if kw_kwargs is None:
    #    keyword_lib = KeywordRegistry()
    #else:
    #    keyword_lib = KeywordRegistry(**kw_kwargs)

    #keyword_lib.register_module(default_keywords)
    #keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))
    #context['registry'] = keyword_lib

    return context


def load_recording_wrapper(load_command=None, exptid="RECORDING", cellid=None,
                           save_cache=True, IsReload=False, modelspecs=None,
                           modelspec=None, **context):
    """
    generic wrapper for loading recordings
    :param load_command: string pointing to relevant load command, eg "my.lib.load_fun"
        load_command should be able to take **context as an input and return
        a dictionary of relevant signals. d : {'stim': X, 'resp': Y, 'state': S} etc.
        X and Y can be in three different forms:
           1. 2D, with M x T and N x T matrices. Ie, the number of channels can differ
              between them but they should have the same times.
           TODO: Support for other stim/resp matrix formats:
           2. 3D, M x T and N x R x T. R corresponds to repetitions of the same X, X will be
              tiled R times to match the length of Y
           3. 4D M x S x T and N x R x S x T or N x S x T. S stimuli were repeated R times
              (or once) X will be unwrapped to M*S x T and tiled if R>1
        additional special dictionary entries:
          d['<signame>_labels'] - list of strings, one to label each row of stim, resp, etc
          d['fs'] - sampling rate for all signals
          d['epochs'] - dataframe with 'start' 'end' 'name' columns for important events.
          d['meta'] - meta data that will be included in rec.meta. regardless, rec.meta
            will be initialized with rec.meta = context
    :param save_cache
        if true the recording will be saved to a NEMS native recording file. context will
        also be used to generate a hash that determines the file name. future calls will
        then generate a hash and load a matching recording from the cache if it exists.
        cached recordings are stored in nems.get_config(NEMS_RECORDINGS_DIR)
        if batch is not None, it will be saved in "<batch>/" subdirectory
        filename will be "<cellid>_<hash>.tgz"
    :param context['batch'] (optional)
        numerical identifier for grouping the recording with other recordings
    :param cellid
        string identifier for the signals being modeled. eg the name of the cell or
        the experiment (if multiple cells are being modeled at once). will extract
        that chan from rec['resp'].chans after loading (but not saved in cache)
    :param context: dictionary of parameters/metadata that will be passed through to load_command

    :return: rec - NEMS recording object

    TODO: option to re-cache
    """
    data_file = recording_filename_hash(exptid, context, nems.get_setting('NEMS_RECORDINGS_DIR'))
    if os.path.exists(data_file):
        log.info("Loading cached file %s", data_file)
        rec = load_recording(data_file)
    else:

        fn = ms._lookup_fn_at(load_command)

        data = fn(exptid=exptid, **context)

        signals = {}
        fs = data.get('fs',100)
        meta = data.get('meta', {})
        epochs = data.get('epochs', None)
        for k in data.keys():
            if k in ['fs', 'meta', 'epochs']:
                pass
            elif k.endswith('_labels'):
                # these wil
                pass
            else:
                signals[k] = RasterizedSignal(fs, data[k], k, exptid, epochs=epochs,
                                              chans=data.get(k+'_labels'))

        rec = Recording(signals)
        rec.meta = meta.copy()
        rec.meta.update(context)
        rec.meta["exptid"] = exptid

        if save_cache:
            rec.save(data_file)

    # if cellid specified, select only that channel
    if cellid is not None:
        if cellid in rec['resp'].chans:
            log.info("match found, extracting channel from rec")
            rec['resp'] = rec['resp'].extract_channels([cellid])
            rec.meta["cellid"] = cellid

    return {'rec': rec}


def load_recordings(recording_uri_list, normalize=False, cellid=None,
                    save_other_cells_to_state=None, input_name='stim',
                    output_name='resp', **context):
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
    if type(cellid) is str:
        if cellid in rec['resp'].chans:
            cellid = [cellid]
        elif len(cellid.split('+')) > 1:
            cellid = cellid.split('+')
        else:
            log.info('No cellid match, keeping all resp channels')

    if cellid is None:
        pass

    elif type(cellid) is list:
        log.info('Extracting channels %s', cellid)
        excluded_cells = [cell for cell in rec['resp'].chans if cell not in cellid]

        if save_other_cells_to_state is not None:
            if type(save_other_cells_to_state) is str:
                pop_var = save_other_cells_to_state
            else:
                pop_var = 'state'
            s = rec['resp'].extract_channels(excluded_cells, name=pop_var).rasterize()
            #s.name = pop_var
            rec.add_signal(s)
            #rec = preproc.concatenate_state_channel(rec, s, pop_var)
            if pop_var == 'state':
                rec['state_raw'] = rec['state']._modified_copy(rec['state']._data, name='state_raw')

        rec['resp'] = rec['resp'].extract_channels(cellid)
    else:
        log.info('No cellid match, keeping all resp channels')

    # Quick fix - will take care of this on the baphy loading side in the future.
    if 'pupil' in rec.signals.keys() and np.any(np.isnan(rec['pupil'].as_continuous())):
                log.info('Padding {0} with the last non-nan value'.format('pupil'))
                inds = ~np.isfinite(rec['pupil'].as_continuous())
                arr = copy.deepcopy(rec['pupil'].as_continuous())
                arr[inds] = arr[~inds][-1]
                rec['pupil'] = rec['pupil']._modified_copy(arr)

    rec = preproc.generate_stim_from_epochs(rec, new_signal_name='epoch_onsets',
                                            epoch_regex='TRIAL', onsets_only=True)

    return {'rec': rec, 'input_name': input_name, 'output_name': output_name}


def normalize_stim(rec=None, sig='stim', norm_method='meanstd', **context):
    """
    Normalize each channel of rec[sig] according to norm_method
    :param rec:  NEMS recording
    :param norm_method:  string {'meanstd', 'minmax'}
    :param context: pass-through for other variables in xforms context dictionary that aren't used.
    :return: copy(?) of rec with updated signal.
    """
    rec[sig] = rec.copy()[sig].rasterize().normalize(norm_method)
    return {'rec': rec}


def init_from_keywords(keywordstring, meta={}, IsReload=False,
                       registry=None, rec=None, input_name='stim',
                       output_name='resp', **context):
    if not IsReload:
        modelspec = init.from_keywords(keyword_string=keywordstring,
                                       meta=meta, registry=registry, rec=rec,
                                       input_name=input_name,
                                       output_name=output_name)
    else:
        modelspec=context['modelspec']
        if modelspec is not None:
            if modelspec.meta.get('input_name', None) is None:
                modelspec.meta['input_name'] = input_name
                modelspec.meta['output_name'] = output_name

    return {'modelspec': modelspec}


def load_modelspecs(modelspecs, uris, IsReload=False, **context):
    '''
    i.e. Load a modelspec from a specific place. This is not
    the same as reloading a model for later inspection; it would be more
    appropriate when doing something complicated with several different
    models.
    '''
    if not IsReload:
        modelspec = ms.ModelSpec([load_resource(uri) for uri in uris])
    return {'modelspec': modelspec}


def set_random_phi(modelspecs, IsReload=False, **context):
    ''' Starts all modelspecs at random phi sampled from the priors. '''
    if not IsReload:
        for fit_idx in modelspec.fit_count:
            modelspec.fit_index = fit_idx
            for i, m in enumerate(modelspec):
                modelspec[i] = priors.set_random_phi(m)
    return {'modelspec': modelspec}


def fill_in_default_metadata(rec, modelspec, IsReload=False, **context):
    '''
    Sets any uninitialized metadata to defaults that should help us
    find it in nems_db again. (fitter, recording, date, etc)
    '''
    if not IsReload:
        # Add metadata to help you reload this state later
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
    return {'modelspec': modelspec}


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
                                    include_incorrect=False, generate_evoked_mask=False,
                                    **context):
    '''
    find REFERENCE epochs spanned by either PASSIVE_EXPERIMENT or
    HIT_TRIAL epochs. mask out all other segments from signals in rec
    '''
    rec = preproc.mask_all_but_correct_references(
            rec, balance_rep_count=balance_rep_count,
            include_incorrect=include_incorrect, generate_evoked_mask=generate_evoked_mask)

    return {'rec': rec}


def mask_all_but_targets(rec, **context):
    '''
    find TARGET epochs all behaviors/outcomes
    '''
    rec = preproc.mask_all_but_targets(rec)

    return {'rec': rec}


def generate_psth_from_resp(rec, epoch_regex='^STIM_', use_as_input=True,
                            smooth_resp=False, **context):
    '''
    generate PSTH prediction from rec['resp'] (before est/val split). Could
    be considered "cheating" b/c predicted PSTH then is based on data in
    val set, but this is because we're interested in testing state effects,
    not sensory coding models. The appropriate control, however is to run
    generate_psth_from_est_for_both_est_and_val_nfold on each nfold est/val
    split.
    '''

    rec = preproc.generate_psth_from_resp(rec, epoch_regex=epoch_regex,
                                          smooth_resp=smooth_resp)
    if use_as_input:
        return {'rec': rec, 'input_name': 'psth'}
    else:
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


def concatenate_input_channels(rec, input_signals=[], **context):
    input_name = context.get('input_name', 'stim')
    rec = preproc.concatenate_input_channels(rec, input_signals=input_signals,
                                             input_name=input_name)

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


def mask_for_jackknife(rec, modelspec=None, epoch_name='REFERENCE',
                       by_time=False, njacks=10, IsReload=False, **context):

    if by_time != True:
        est_out, val_out, modelspec_out = \
            preproc.mask_est_val_for_jackknife(rec, modelspec=modelspec,
                                               epoch_name=epoch_name,
                                               njacks=njacks, IsReload=IsReload)
    else:
        est_out, val_out, modelspec_out = \
            preproc.mask_est_val_for_jackknife_by_time(rec, modelspec=modelspec,
                                               njacks=njacks, IsReload=IsReload)

    if IsReload:
        return {'est': est_out, 'val': val_out}
    else:
        return {'est': est_out, 'val': val_out,
                'jackknifed_fit': True, 'modelspec': modelspec_out}


def jack_subset(est, val, modelspec=None, IsReload=False,
                keep_only=1, **context):

    if keep_only == 1:
        est = est.views(view_range=0)[0]
        val = val.views(view_range=0)[0]
        est['resp']=est['resp'].rasterize()
        val['resp']=val['resp'].rasterize()
        est['stim']=est['stim'].rasterize()
        val['stim']=val['stim'].rasterize()

    else:
        est = est.views(keep_only)[0]
        val = val.views(keep_only)[0]

    if modelspec is not None:
        modelspec_out = modelspec.copy(fit_index=keep_only)
        modelspec_out.fit_index = 0

    if IsReload:
        return {'est': est, 'val': val, 'jackknifed_fit': False}
    else:
        return {'est': est, 'val': val, 'modelspec': modelspec_out, 'jackknifed_fit': False}


###############################################################################
######################        INITIALIZERS         ############################
###############################################################################


def fit_basic_init(modelspec, est, tolerance=10**-5.5, metric='nmse',
                   IsReload=False, norm_fir=False, nl_kw={},
                   jackknifed_fit=False, output_name='resp', **context):
    '''
    Initialize modelspecs in a way that avoids getting stuck in
    local minima.

    written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)
    '''
    # only run if fitting
    if IsReload:
        return {}

    if isinstance(metric, str):
        metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name)
    else:
        metric_fn = metric

    # TODO : merge JK and non-JK code if possible.
    if jackknifed_fit:
        nfolds = est.view_count
        if modelspec.jack_count < est.view_count:
            modelspec.tile_jacks(nfolds)
            # TODO replace with tile_jacks
            #  allow nested loop for multiple fits (init conditions) within a jackknife
            #  initialize each jackknife with the same random ICs?
        for fit_idx in range(modelspec.fit_count):
            for jack_idx, e in enumerate(est.views()):
                modelspec.fit_index = fit_idx
                log.info("----------------------------------------------------")
                log.info("Init fitting model fit %d/%d, fold %d/%d",
                         fit_idx+1, modelspec.fit_count,
                         jack_idx + 1, modelspec.jack_count)

                modelspec = nems.initializers.prefit_LN(
                        e, modelspec.set_jack(jack_idx),
                        analysis_function=nems.analysis.api.fit_basic,
                        fitter=scipy_minimize, metric=metric_fn,
                        tolerance=tolerance, max_iter=700, norm_fir=norm_fir,
                        nl_kw=nl_kw)
    else:
        #import pdb
        #pdb.set_trace()
        for fit_idx in range(modelspec.fit_count):
            log.info("Init fitting model instance %d/%d", fit_idx + 1, modelspec.fit_count)
            modelspec = nems.initializers.prefit_LN(
                    est, modelspec.set_fit(fit_idx),
                    analysis_function=nems.analysis.api.fit_basic,
                    fitter=scipy_minimize, metric=metric_fn,
                    tolerance=tolerance, max_iter=700, norm_fir=norm_fir,
                    nl_kw=nl_kw)

    return {'modelspec': modelspec}


def _set_zero(x):
    """ fill x with zeros, except preserve nans """
    y = x.copy()
    y[np.isfinite(y)] = 0
    return y


def fit_state_init(modelspec, est, tolerance=10**-5.5, metric='nmse',
                   IsReload=False, norm_fir=False, nl_kw = {},
                   fit_sig='resp', output_name='resp', **context):

    '''
    Initialize modelspecs in an attempt to avoid getting stuck in
    local minima. Remove state replication/merging first.

    written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)

    assumption -- est['state'] signal is being used for merge
    '''
    if not IsReload:
        metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name)

        for fit_idx in range(modelspec.fit_count):
            for jack_idx, d in enumerate(est.views()):
                modelspec.fit_index = fit_idx
                modelspec.jack_index = jack_idx
                log.info("----------------------------------------------------")
                log.info("Init state-free fitting model fit %d/%d, fold %d/%d",
                         fit_idx+1, modelspec.fit_count,
                         jack_idx + 1, modelspec.jack_count)

                # set state to 0 for all timepoints so that only first filterbank
                # is used
                dc = d.copy()
                dc['state'] = dc['state'].transform(_set_zero, 'state')
                if fit_sig != 'resp':
                    log.info("Subbing %s for resp signal", fit_sig)
                    dc['resp'] = dc[fit_sig]

                modelspec = nems.initializers.prefit_LN(
                        dc, modelspec,
                        analysis_function=nems.analysis.api.fit_basic,
                        fitter=scipy_minimize, metric=metric_fn,
                        tolerance=tolerance, max_iter=700, norm_fir=norm_fir,
                        nl_kw=nl_kw)
                # fit a bit more to settle in STP variables and anything else
                # that might have been excluded
                fit_kwargs = {'tolerance': tolerance/2, 'max_iter': 500}
                modelspec = nems.analysis.api.fit_basic(
                        dc, modelspec, fit_kwargs=fit_kwargs, metric=metric_fn,
                        fitter=scipy_minimize)
                rep_idx = find_module('replicate_channels', modelspec)
                mrg_idx = find_module('merge_channels', modelspec)
                if rep_idx is not None:
                    repcount = modelspec[rep_idx]['fn_kwargs']['repcount']
                    for j in range(rep_idx+1, mrg_idx):
                        # assume all phi
                        log.debug(modelspec[j]['fn'])
                        if 'phi' in modelspec[j].keys():
                            for phi in modelspec[j]['phi'].keys():
                                s = modelspec[j]['phi'][phi].shape
                                setcount = int(s[0] / repcount)
                                log.debug('phi[%s] setcount=%d', phi, setcount)
                                snew = np.ones(len(s))
                                snew[0] = repcount
                                new_v = np.tile(modelspec[j]['phi'][phi][:setcount, ...],
                                                snew.astype(int))
                                log.debug(new_v)
                                modelspec[j]['phi'][phi] = new_v

    return {'modelspec': modelspec}


###############################################################################
########################       FITTERS / ANALYSES      ########################
###############################################################################


def fit_basic(modelspec, est, max_iter=1000, tolerance=1e-7,
              metric='nmse', IsReload=False, fitter='scipy_minimize',
              jackknifed_fit=False, random_sample_fit=False,
              n_random_samples=0, random_fit_subset=None,
              output_name='resp', **context):
    ''' A basic fit that optimizes every input modelspec. '''

    if IsReload:
        return {}
    metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name)
    fitter_fn = getattr(nems.fitters.api, fitter)
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    if modelspec.jack_count < est.view_count:
        raise Warning('modelspec.jack_count does not match est.view_count')
        modelspec.tile_jacks(nfolds)
    for fit_idx in range(modelspec.fit_count):
        for jack_idx, e in enumerate(est.views()):
            modelspec.jack_index = jack_idx
            modelspec.fit_index = fit_idx
            log.info("----------------------------------------------------")
            log.info("Fitting: fit %d/%d, fold %d/%d",
                     fit_idx + 1, modelspec.fit_count,
                     jack_idx + 1, modelspec.jack_count)
            modelspec = nems.analysis.api.fit_basic(
                    e, modelspec, fit_kwargs=fit_kwargs,
                    metric=metric_fn, fitter=fitter_fn)

    return {'modelspec': modelspec}


def reverse_correlation(modelspec, est, IsReload=False, jackknifed_fit=False,
                input_name='stim', output_name='resp', **context):
    ''' Perform basic normalized reverse correlation between input '''

    if not IsReload:

        if jackknifed_fit:
            nfolds = est.view_count
            if modelspec.jack_count < est.view_count:
                modelspec.tile_jacks(nfolds)
            for fit_idx in range(modelspec.fit_count):
                for jack_idx, e in enumerate(est.views()):
                    modelspec.fit_index = fit_idx
                    modelspec.jack_index = jack_idx
                    log.info("----------------------------------------------------")
                    log.info("Fitting: fit %d/%d, fold %d/%d",
                             fit_idx + 1, modelspec.fit_count,
                             jack_idx + 1, modelspec.jack_count)

                    modelspec = nems.analysis.api.reverse_correlation(
                            e, modelspec, input_name)

        else:
            # standard single shot
            for fit_idx in range(modelspec.fit_count):
                modelspec = nems.analysis.api.reverse_correlation(
                    est, modelspec.set_fit(fit_idx), input_name)

    return {'modelspec': modelspec}


def fit_iteratively(modelspec, est, tol_iter=100, fit_iter=20, IsReload=False,
                    module_sets=None, invert=False, tolerances=[1e-4, 1e-5, 1e-6, 1e-7],
                    metric='nmse', fitter='scipy_minimize', fit_kwargs={},
                    jackknifed_fit=False, output_name='resp', **context):
    if IsReload:
        return {}

    fitter_fn = getattr(nems.fitters.api, fitter)
    metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name)

    if modelspec.jack_count < est.view_count:
        raise Warning('modelspec.jack_count does not match est.view_count')
        modelspec.tile_jacks(nfolds)

    for fit_idx in range(modelspec.fit_count):
        for jack_idx, e in enumerate(est.views()):
            modelspec.jack_index = jack_idx
            modelspec.fit_index = fit_idx
            log.info("----------------------------------------------------")
            log.info("Iter fitting: fit %d/%d, fold %d/%d",
                     fit_idx + 1, modelspec.fit_count,
                     jack_idx + 1, modelspec.jack_count)
            modelspec = nems.analysis.api.fit_iteratively(
                        e, modelspec, fit_kwargs=fit_kwargs,
                        fitter=fitter_fn, module_sets=module_sets,
                        invert=invert, tolerances=tolerances,
                        tol_iter=tol_iter, fit_iter=fit_iter,
                        metric=metric_fn)

    return {'modelspec': modelspec}


def fit_nfold(modelspecs, est, tolerance=1e-7, max_iter=1000,
              IsReload=False, metric='nmse', fitter='scipy_minimize',
              analysis='fit_basic', tolerances=None, module_sets=None,
              tol_iter=100, fit_iter=20, output_name='resp', **context):
    ''' fitting n fold, one from each entry in est '''
    raise Warning("DEPRECATED?")
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
    raise Warning ('This is deprecated. Replaced by set_random_phi and analysis.test_prediction.pick_best_phi')
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


def save_recordings(modelspec, est, val, **context):
    # TODO: Save the recordings somehow?
    return {'modelspec': modelspec}


def predict(modelspec, est, val, jackknifed_fit=False, **context):
    # modelspecs = metrics.add_summary_statistics(est, val, modelspecs)
    # TODO: Add statistics to metadata of every modelspec

    est, val = nems.analysis.api.generate_prediction(est, val, modelspec, jackknifed_fit=jackknifed_fit)
    modelspec.recording = val

    return {'val': val, 'est': est, 'modelspec': modelspec}


def add_summary_statistics(est, val, modelspec, fn='standard_correlation',
                           rec=None, use_mask=True, **context):
    '''
    standard_correlation: average all correlation metrics and add
                          to first modelspec only.
    correlation_per_model: evaluate correlation metrics separately for each
                           modelspec and save results in each modelspec
    '''
    corr_fn = getattr(nems.analysis.api, fn)
    modelspec = corr_fn(est, val, modelspec=modelspec, rec=rec, use_mask=use_mask)

    if find_module('state', modelspec) is not None:
        s = metrics.state_mod_index(val, epoch='REFERENCE', psth_name='pred',
                            state_sig='state_raw', state_chan=[])
        j_s, ee = metrics.j_state_mod_index(val, epoch='REFERENCE', psth_name='pred',
                            state_sig='state_raw', state_chan=[], njacks=10)
        modelspec.meta['state_mod'] = s
        modelspec.meta['j_state_mod'] = j_s
        modelspec.meta['se_state_mod'] = ee
        modelspec.meta['state_chans'] = val['state'].chans

        # Charlie testing diff ways to calculate mod index

        # try using resp
        s = metrics.state_mod_index(val, epoch='REFERENCE', psth_name='resp',
                            state_sig='state_raw', state_chan=[])
        j_s, ee = metrics.j_state_mod_index(val, epoch='REFERENCE', psth_name='resp',
                            state_sig='state_raw', state_chan=[], njacks=10)
        modelspec.meta['state_mod_r'] = s
        modelspec.meta['j_state_mod_r'] = j_s
        modelspec.meta['se_state_mod_r'] = ee

        # try using the "mod" signal (if it exists) which is calculated
        if 'mod' in modelspec.meta['modelname']:
            s = metrics.state_mod_index(val, epoch='REFERENCE',
                                            psth_name='mod', divisor='resp',
                                            state_sig='state_raw', state_chan=[])
            j_s, ee = metrics.j_state_mod_index(val, epoch='REFERENCE',
                                            psth_name='mod', divisor='resp',
                                            state_sig='state_raw', state_chan=[],
                                            njacks=10)
            modelspec.meta['state_mod_m'] = s
            modelspec.meta['j_state_mod_m'] = j_s
            modelspec.meta['se_state_mod_m'] = ee

    return {'modelspec': modelspec}


def plot_summary(modelspec, val, figures=None, IsReload=False,
                 figures_to_load=None, **context):
    # CANNOT initialize figures=[] in optional args our you will create a bug

    if figures is None:
        figures = []
    if not IsReload:
        fig = nplt.quickplot({'modelspec': modelspec, 'val': val})
        # Needed to make into a Bytes because you can't deepcopy figures!
        figures.append(nplt.fig2BytesIO(fig))
    else:
        if figures_to_load is not None:
            figures.extend([load_resource(f) for f in figures_to_load])

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

def fast_eval(modelspec, est, **context):
    modelspec.fast_eval_on(est)

    return {'modelspec': modelspec}


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


def save_analysis(destination, recording, modelspec, xfspec, figures,
                  log, add_tree_path=False):
    '''Save an analysis file collection to a particular destination.'''
    if add_tree_path:
        treepath = tree_path(recording, [modelspec], xfspec)
        base_uri = os.path.join(destination, treepath)
    else:
        base_uri = destination

    if destination is None:
        destination = get_default_savepath(modelspec)
        base_uri = destination

    modelspec.meta['modelpath'] = base_uri
    modelspec.meta['figurefile'] = os.path.join(base_uri,'figure.0000.png')
    base_uri = base_uri if base_uri[-1] == '/' else base_uri + '/'
    xfspec_uri = base_uri + 'xfspec.json'  # For attaching to modelspecs

    for number in range(modelspec.jack_count):
        m = modelspec.copy()
        m.jack_index = number
        set_modelspec_metadata(m, 'xfspec', xfspec_uri)
        save_resource(base_uri + 'modelspec.{:04d}.json'.format(number), json=m[:])
    for number, figure in enumerate(figures):
        save_resource(base_uri + 'figure.{:04d}.png'.format(number), data=figure)
    save_resource(base_uri + 'log.txt', data=log)
    save_resource(xfspec_uri, json=xfspec)
    return {'savepath': base_uri}


def load_analysis(filepath, eval_model=True, only=None):
    """
    load xforms spec and context dictionary for a model fit
    :param filepath: URI of saved xforms model
    :param eval_model: if True, re-evaluates all steps. Time consuming but gives an exact copy of the
    original context
    :param only: either an int, usually 0, to evaluate the first step of loading a recording, or a slice
    object, which gives more flexibility over what steps of the original xfspecs to run again.
    :return: (xfspec, ctx) tuple
    """
    log.info('Loading xfspec and context from %s...', filepath)

    xfspec = load_xform(os.path.join(filepath, 'xfspec.json'))
    mspaths = []
    figures_to_load = []
    logstring = ''
    for file in os.listdir(filepath):
        if file.startswith("modelspec"):
            mspaths.append(os.path.join(filepath, file))
        elif file.startswith("figure"):
            figures_to_load.append(os.path.join(filepath, file))
        elif file.startswith("log"):
            logpath = os.path.join(filepath, file)
            with open(logpath) as logfile:
                logstring = logfile.read()
    ctx = load_modelspecs([], uris=mspaths, IsReload=False)
    ctx['IsReload'] = True
    ctx['figures_to_load'] = figures_to_load
    ctx['log'] = logstring

    if eval_model:
        ctx, log_xf = evaluate(xfspec, ctx)
    elif only is not None:
        # Useful for just loading the recording without doing
        # any subsequent evaluation.
        if isinstance(only, int):
            ctx, log_xf = evaluate([xfspec[only]], ctx)
        elif isinstance(only, slice):
            ctx, log_xf = evaluate(xfspec[only], ctx)

    return xfspec, ctx


def regenerate_figures(batch, modelnames, cellids=None):
    '''
    Regenerate quickplot figures for a given modelname and batch.
    Intended to be used in cases where plot code has been updated and
    needs to be included in existing models, but a re-fit is not necessary.

    '''
    if cellids is None:
        cells = nd.get_batch_cells(batch)['cellid'].values.tolist()
    else:
        cells = cellids
    r = nd.get_results_file(batch, modelnames, cellids=cellids)

    for m in modelnames:
        for c in cells:
            try:
                filepath = r[r.cellid == c][r.modelname == m]['modelpath'].iat[0] + '/'
            except IndexError:
                # result doesn't exist
                continue
            xfspec, ctx = load_analysis(filepath, eval_model=True)
            fig = nplt.quickplot(ctx)
            fig_bytes = nplt.fig2BytesIO(fig)
            plt.close('all')
            save_resource(filepath + 'figure.0000.png', data=fig_bytes)
            # TODO: also have to tell


###############################################################################
##################        CONTEXT UTILITIES             #######################
###############################################################################


def evaluate_context(ctx, rec_key='val', rec_idx=0, mspec_idx=0, start=None,
                     stop=None):
    rec = ctx[rec_key][0]
    mspec = ctx['modelspec']
    mspec.fit_index = mspec_idx
    return ms.evaluate(rec, mspec, start=start, stop=stop)


def get_meta(ctx, mspec_idx=0, mod_idx=0):
    if (mspec_idx > 0) or (mod_idx >0):
        raise ValueError("meta indexing not supported")
    return ctx['modelspec'].meta


def get_modelspec(ctx, mspec_idx=0):
    m = ctx['modelspec'].copy()
    m.fit_index = mspec_idx
    return m


def get_module(ctx, val, key='index', mspec_idx=0, find_all_matches=False):
    mspec = ctx['modelspec']
    mspec.fit_index = mspec_idx
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
