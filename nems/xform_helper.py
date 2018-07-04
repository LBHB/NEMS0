import logging
import importlib as imp

import nems.xforms as xforms
from nems import get_setting
from nems.registry import KeywordRegistry
from nems.plugins import default_keywords
from nems.plugins import default_loaders
from nems.plugins import default_fitters

log = logging.getLogger(__name__)


def generate_xforms_spec(recording_uri, modelname, meta={}, autoPlot=True):
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
    load_keywords, model_keywords, fit_keywords = modelname.split("_")

    loader_lib = KeywordRegistry(recording_uri)
    loader_lib.register_module(default_loaders)
    loader_lib.register_plugins(get_setting('XF_LOADER_PLUGINS'))
    loader_xfspec = []
    for loadkey in load_keywords.split('-'):
        loader_xfspec.extend(loader_lib[loadkey])

    fitter_lib = KeywordRegistry()
    fitter_lib.register_module(default_fitters)
    fitter_lib.register_plugins(get_setting('XF_FITTER_PLUGINS'))
    fitter_xfspec = []
    for fitkey in fit_keywords.split('-'):
        fitter_xfspec = fitter_lib[fitkey]

    keyword_lib = KeywordRegistry()
    keyword_lib.register_module(default_keywords)
    keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

    # Generate the xfspec, which defines the sequence of events
    # to run through (like a packaged-up script)
    xfspec = []

    # 1) Load the data
    xfspec.extend(loader_xfspec)

    # 2) generate a modelspec
    xfspec.append(['nems.xforms.init_from_keywords',
                   {'keywordstring': model_keywords, 'meta': meta,
                    'registry': keyword_lib}])

    # 3) fit the data
    xfspec.extend(fitter_xfspec)

    # 4) add some performance statistics
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

    # 5) generate plots
    if autoPlot:
        log.info('Adding summary plot to xfspec...')
        xfspec.append(['nems.xforms.plot_summary', {}])

    return xfspec


def fit_xfspec(xfspec):
    # Now that the xfspec is assembled, run through it
    # in order to get the fitted modelspec, evaluated recording, etc.
    # (all packaged up in the ctx dictionary).
    ctx, log_xf = xforms.evaluate(xfspec)
    return ctx
