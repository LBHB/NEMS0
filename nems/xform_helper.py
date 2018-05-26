import logging
import importlib as imp

import nems.xforms as xforms
from nems import get_setting
from nems.registry import KeywordRegistry
from nems.plugins.loaders import default_loaders
from nems.plugins.fitters import default_fitters

log = logging.getLogger(__name__)


def fit_model_xforms(recording_uri, modelname, autoPlot=True):
    """
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
    kws = modelname.split("_")
    loadkey = kws[0]
    fitkey = kws[-1]

    loader_lib = KeywordRegistry(recording_uri)
    loader_lib.register_module(default_loaders)
    loader_lib.register_plugins(get_setting('XF_LOADER_PLUGINS'))
    loader_xfspec = loader_lib[loadkey]

    fitter_lib = KeywordRegistry()
    fitter_lib.register_module(default_fitters)
    fitter_lib.register_plugins(get_setting('XF_FITTER_PLUGINS'))
    fitter_xfspec = fitter_lib[fitkey]

    modelspecname = "_".join(kws[1:-1])

    meta = {'modelname': modelname, 'loader': loadkey, 'fitkey': fitkey,
            'modelspecname': modelspecname}

    # TODO: These should be added to meta by nems_db after ctx is returned.
    #       'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
    #       'githash': os.environ.get('CODEHASH', ''),
    #       'recording': loader}

    # Generate the xfspec, which defines the sequence of events
    # to run through (like a packaged-up script)
    xfspec = []

    # 1) Load the data
    xfspec.extend(loader_xfspec)

    # 2) generate a modelspec
    kw_module = imp.import_module(get_setting('KW_REGISTRY_MODULE'))
    kw_registry = getattr(kw_module, get_setting('KW_REGISTRY_NAME'))
    xfspec.append(['nems.xforms.init_from_keywords',
                   {'keywordstring': modelspecname, 'meta': meta,
                    'registry': kw_registry}])

    # 3) fit the data
    xfspec.extend(fitter_xfspec)

    # 4) add some performance statistics
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

    # 5) generate plots
    if autoPlot:
        log.info('Generating summary plot...')
        xfspec.append(['nems.xforms.plot_summary', {}])

    # Now that the xfspec is assembled, run through it
    # in order to get the fitted modelspec, evaluated recording, etc.
    # (all packaged up in the ctx dictionary).
    ctx, log_xf = xforms.evaluate(xfspec)

    return ctx
