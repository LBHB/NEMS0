"""xforms_helper library

This module contains wrappers for transformations ("xforms") applied sequentially during a NEMS
fitting process.

"""
import logging
import os

import nems.xforms as xforms
import nems.db as nd
from nems import get_setting
from nems.utils import escaped_split, escaped_join
from nems.registry import KeywordRegistry
from nems.plugins import (default_keywords, default_loaders, default_fitters,
                          default_initializers)

log = logging.getLogger(__name__)


def generate_xforms_spec(recording_uri=None, modelname=None, meta={},
                         xforms_kwargs={}, xforms_init_context=None,
                         kw_kwargs={}, autoPred=True, autoStats=True,
                         autoPlot=True):
    """
    Generate an xforms spec based on a modelname, which can then be evaluated
    in order to process and fit a model.

    Parameter
    ---------
    recording_uri : str
        Location to load recording from, e.g. a filepath or URL.
    modelname : str
        NEMS-formatted modelname, e.g. 'ld-sev_wc.18x2-fir.2x15-dexp.1_basic'
        The modelname will be parsed into a series of xforms functions using
        xforms and keyword registries.
    meta : dict
        Additional keyword arguments for nems.initializers.init_from_keywords
    xforms_kwargs : dict
        Additional keyword arguments for the xforms registry. DEPRECATED??
    kw_kwargs : dict
        Additional keyword arguments for the keyword registry. DEPRECATED?
    xforms_init_context : dict
        Initialization for context. REPLACES xforms_kwargs and kw_kwargs???
    autoPred : boolean
        If true, will automatically append nems.xforms.predict to the xfspec
        if it is not already present.
    autoStats : boolean
        If true, will automatically append nems.xforms.add_summary_statistics
        to the xfspec if it is not already present.
    autoPlot : boolean
        If true, will automatically append nems.xforms.plot_summary to the
        xfspec if it is not already present.

    Returns
    -------
    xfspec : list of 2- or 4- tuples

    """

    log.info('Initializing modelspec(s) for recording/model {0}/{1}...'
             .format(recording_uri, modelname))

    # parse modelname and assemble xfspecs for loader and fitter

    # TODO: naming scheme change: pre_modules, modules, post_modules?
    #       or something along those lines... since they aren't really
    #       just loaders and fitters
    load_keywords, model_keywords, fit_keywords = escaped_split(modelname, '_')
    if recording_uri is not None:
        xforms_lib = KeywordRegistry(recording_uri=recording_uri, **xforms_kwargs)
    else:
        xforms_lib = KeywordRegistry(**xforms_kwargs)

    xforms_lib.register_modules([default_loaders, default_fitters,
                                 default_initializers])
    xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

    keyword_lib = KeywordRegistry()
    keyword_lib.register_module(default_keywords)
    keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

    # Generate the xfspec, which defines the sequence of events
    # to run through (like a packaged-up script)
    xfspec = []

    # 0) set up initial context
    if xforms_init_context is None:
        xforms_init_context = {}
    if kw_kwargs is not None:
         xforms_init_context['kw_kwargs'] = kw_kwargs
    xforms_init_context['keywordstring'] = model_keywords
    xforms_init_context['meta'] = meta
    xfspec.append(['nems.xforms.init_context', xforms_init_context])

    # 1) Load the data
    xfspec.extend(_parse_kw_string(load_keywords, xforms_lib))

    # 2) generate a modelspec
    xfspec.append(['nems.xforms.init_from_keywords', {'registry': keyword_lib}])
    #xfspec.append(['nems.xforms.init_from_keywords', {}])

    # 3) fit the data
    xfspec.extend(_parse_kw_string(fit_keywords, xforms_lib))

    # TODO: need to make this smarter about how to handle the ordering
    #       of pred/stats when only stats is overridden.
    #       For now just have to manually include pred if you want to
    #       do your own stats or plot xform (like using stats.pm)

    # 4) generate a prediction (optional)
    if autoPred:
        if not _xform_exists(xfspec, 'nems.xforms.predict'):
            xfspec.append(['nems.xforms.predict', {}])

    # 5) add some performance statistics (optional)
    if autoStats:
        if not _xform_exists(xfspec, 'nems.xforms.add_summary_statistics'):
            xfspec.append(['nems.xforms.add_summary_statistics', {}])

    # 6) generate plots (optional)
    if autoPlot:
        if not _xform_exists(xfspec, 'nems.xforms.plot_summary'):
            # log.info('Adding summary plot to xfspec...')
            xfspec.append(['nems.xforms.plot_summary', {}])

    return xfspec


def fit_xfspec(xfspec):
    # Now that the xfspec is assembled, run through it
    # in order to get the fitted modelspec, evaluated recording, etc.
    # (all packaged up in the ctx dictionary).
    ctx, log_xf = xforms.evaluate(xfspec)
    return ctx


def fit_model_xform(cellid, batch, modelname, autoPlot=True, saveInDB=False,
                    returnModel=False):
    """
    Fit a single NEMS model using data stored in database. First generates an xforms
    script based on modelname parameter and then evaluates it.
    :param cellid: cellid and batch specific dataset in database
    :param batch:
    :param modelname: string specifying model architecture, preprocessing
    and fit method
    :param autoPlot: generate summary plot when complete
    :param saveInDB: save results to Results table
    :param returnModel: boolean (default False). If False, return savepath
       if True return xfspec, ctx tuple
    :return: savepath = path to saved results or (xfspec, ctx) tuple
    """

    log.info('Initializing modelspec(s) for cell/batch %s/%d...',
             cellid, int(batch))

    # Segment modelname for meta information
    kws = escaped_split(modelname, '_')

    modelspecname = escaped_join(kws[1:-1], '-')
    loadkey = kws[0]
    fitkey = kws[-1]

    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loadkey, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loadkey}

    if type(cellid) is list:
        meta['siteid'] = cellid[0][:7]

    # registry_args = {'cellid': cellid, 'batch': int(batch)}
    registry_args = {}
    xforms_init_context = {'cellid': cellid, 'batch': int(batch)}

    log.info("TODO: simplify generate_xforms_spec parameters")
    xfspec = generate_xforms_spec(recording_uri=None, modelname=modelname,
                                  meta=meta,  xforms_kwargs=registry_args,
                                  xforms_init_context=xforms_init_context,
                                  autoPlot=autoPlot)
    log.info(xfspec)

    # actually do the loading, preprocessing, fit
    ctx, log_xf = xforms.evaluate(xfspec)

    # save some extra metadata
    modelspec = ctx['modelspec']

    # figure out URI for location to save results (either file or http, depending on USE_NEMS_BAPHY_API)
    if get_setting('USE_NEMS_BAPHY_API'):
        prefix = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT')) + '/results/'
    else:
        prefix = get_setting('NEMS_RESULTS_DIR')

    if type(cellid) is list:
        cell_name = cellid[0].split("-")[0]
    else:
        cell_name = cellid

    destination = os.path.join(prefix, str(batch), cell_name, modelspec.get_longname())

    modelspec.meta['modelpath'] = destination
    modelspec.meta['figurefile'] = os.path.join(destination, 'figure.0000.png')
    modelspec.meta.update(meta)

    if returnModel:
        # return fit, skip save!
        return xfspec, ctx

    # save results
    log.info('Saving modelspec(s) to {0} ...'.format(destination))
    if 'figures' in ctx.keys():
        figs = ctx['figures']
    else:
        figs = []
    save_data = xforms.save_analysis(destination,
                                     recording=ctx['rec'],
                                     modelspec=modelspec,
                                     xfspec=xfspec,
                                     figures=figs,
                                     log=log_xf)

    # save in database as well
    if saveInDB:
        nd.update_results_table(modelspec)

    return save_data['savepath']


def load_model_xform(cellid, batch=271,
        modelname="ozgf100ch18_wcg18x2_fir15x2_lvl1_dexp1_fit01",
        eval_model=True, only=None):
    '''
    Load a model that was previously fit via fit_model_xforms

    Parameters
    ----------
    cellid : str
        cellid in celldb database
    batch : int
        batch number in celldb database
    modelname : str
        modelname in celldb database
    eval_model : boolean
        If true, the entire xfspec will be re-evaluated after loading.
    only : int
        Index of single xfspec step to evaluate if eval_model is False.
        For example, only=0 will typically just load the recording.

    Returns
    -------
    xfspec, ctx : nested list, dictionary

    '''

    kws = escaped_split(modelname, '_')
    old = False
    if (len(kws) > 3) or ((len(kws) == 3) and kws[1].startswith('stategain')
                          and not kws[1].startswith('stategain.')):
        # Check if modelname uses old format.
        log.info("Using old modelname format ... ")
        old = True

    d = nd.get_results_file(batch, [modelname], [cellid])
    filepath = d['modelpath'][0]

    if old:
        raise NotImplementedError("need to use oxf library.")
        xfspec, ctx = oxf.load_analysis(filepath, eval_model=eval_model)
    else:
        xfspec, ctx = xforms.load_analysis(filepath, eval_model=eval_model,
                                           only=only)
    return xfspec, ctx


def _parse_kw_string(kw_string, registry):
    xfspec = []
    for kw in escaped_split(kw_string, '-'):
        try:
            xfspec.extend(registry[kw])
        except KeyError:
            log.error("No keyword found for: '%s', raising KeyError.", kw)
            raise

    return xfspec


def _xform_exists(xfspec, xform_fn):
    for xf in xfspec:
        if xf[0] == xform_fn:
            return True
