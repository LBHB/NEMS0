import logging
import importlib as imp

import nems.xforms as xforms
from nems import get_setting
from nems.utils import escaped_split
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
        Additional keyword arguments for the xforms registry
    kw_kwargs : dict
        Additional keyword arguments for the keyword registry
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

    keyword_lib = KeywordRegistry(**kw_kwargs)
    keyword_lib.register_module(default_keywords)
    keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

    # Generate the xfspec, which defines the sequence of events
    # to run through (like a packaged-up script)
    xfspec = []

    # 0) set up initial context
    if xforms_init_context is None:
        xforms_init_context = {}
    xforms_init_context['keywordstring'] = model_keywords
    xforms_init_context['meta'] = meta
    xfspec.append(['nems.xforms.init_context', xforms_init_context])

    # 1) Load the data
    xfspec.extend(_parse_kw_string(load_keywords, xforms_lib))

    # 2) generate a modelspec
    xfspec.append(['nems.xforms.init_from_keywords', {'registry': keyword_lib}])

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


def _parse_kw_string(kw_string, registry):
    xfspec = []
    for kw in escaped_split(kw_string, '-'):
        try:
            xfspec.extend(registry[kw])
        except KeyError:
            log.info("No keyword found for: %s , skipping ...", kw)
            pass

    return xfspec


def _xform_exists(xfspec, xform_fn):
    for xf in xfspec:
        if xf[0] == xform_fn:
            return True
