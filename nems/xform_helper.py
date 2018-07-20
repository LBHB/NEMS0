import logging
import importlib as imp

import nems.xforms as xforms
from nems import get_setting
from nems.registry import KeywordRegistry
from nems.plugins import (default_keywords, default_loaders, default_fitters,
                          default_initializers)

log = logging.getLogger(__name__)


def generate_xforms_spec(recording_uri, modelname, meta={}, autoPred=True,
                         autoStats=True, autoPlot=True):
    """
    TODO: Update this doc

    OUTDATED
    Fits a single NEMS model
    eg, 'ozgf100ch18_wc18x1_lvl1_fir15x1_dexp1_fit01'
    generates modelspec with 'wc18x1_lvl1_fir1x15_dexp1'

    based on fit_model function in nems/scripts/fit_model.py

    example xfspec:
     xfspec = [
        ['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
        ['nems.xforms.add_average_sig', {'signal_to_average': 'resp',
                                         'new_signalname': 'resp',
                                         'epoch_regex': '^STIM_'}],
        ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
        ['nems.xforms.init_from_keywords', {'keywordstring': modelspecname}],
        ['nems.xforms.set_random_phi',  {}],
        ['nems.xforms.fit_basic',       {}],
        # ['nems.xforms.add_summary_statistics',    {}],
        ['nems.xforms.plot_summary',    {}],
        # ['nems.xforms.save_recordings', {'recordings': ['est', 'val']}],
        ['nems.xforms.fill_in_default_metadata',    {}],
     ]
    """

    log.info('Initializing modelspec(s) for recording/model {0}/{1}...'
             .format(recording_uri, modelname))

    # parse modelname and assemble xfspecs for loader and fitter

    # TODO: naming scheme change: pre_modules, modules, post_modules?
    #       or something along those lines... since they aren't really
    #       just loaders and fitters
    load_keywords, model_keywords, fit_keywords = modelname.split("_")

    xforms_lib = KeywordRegistry(recording_uri=recording_uri)
    xforms_lib.register_modules([default_loaders, default_fitters,
                                 default_initializers])
    xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

    keyword_lib = KeywordRegistry()
    keyword_lib.register_module(default_keywords)
    keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

    # Generate the xfspec, which defines the sequence of events
    # to run through (like a packaged-up script)
    xfspec = []

    # 1) Load the data
    xfspec.extend(_parse_kw_string(load_keywords, xforms_lib))

    # 2) generate a modelspec
    xfspec.append(['nems.xforms.init_from_keywords',
                   {'keywordstring': model_keywords, 'meta': meta,
                    'registry': keyword_lib}])

    # 3) fit the data
    xfspec.extend(_parse_kw_string(fit_keywords, xforms_lib))

    # TODO: need to make this smarter about how to handle the ordering
    #       of pred/stats when only stats is overridden.

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
            log.info('Adding summary plot to xfspec...')
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
    for kw in kw_string.split('-'):
        try:
            xfspec.extend(registry[kw])
        except KeyError:
            log.info("No keyword found for: %s , skipping ...")
            pass

    return xfspec


def _xform_exists(xfspec, xform_fn):
    for xf in xfspec:
        if xf[0] == xform_fn:
            return True
