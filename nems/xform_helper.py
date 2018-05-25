import logging
import importlib as imp

import nems.xforms as xforms
from nems import get_setting

log = logging.getLogger(__name__)


def default_loader_xfspec(loadkey, recording_uri):
    return None


def ozgf(loadkey, recording_uri):
    recordings = [recording_uri]

    if loadkey in ["ozgf100ch18", "ozgf100ch18n"]:
        normalize = int(loader == "ozgf100ch18n")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.split_by_occurrence_counts',
                   {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences', {}]]

    elif loadkey in ["ozgf100ch18pup", "ozgf100ch18npup"]:
        normalize = int(loader == "ozgf100ch18npup")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.make_state_signal',
                   {'state_signals': ['pupil'], 'permute_signals': [],
                    'new_signalname': 'state'}]]

    return xfspec


def _state_model_loadkey_helper(loadkey):

    if loadkey.endswith("beh0"):
        state_signals = ['active']
        permute_signals = ['active']
    elif loadkey.endswith("beh"):
        state_signals = ['active']
        permute_signals = []
    elif loadkey.endswith("pup0beh0"):
        state_signals = ['pupil', 'active']
        permute_signals = ['pupil', 'active']
    elif loadkey.endswith("pup0beh"):
        state_signals = ['pupil', 'active']
        permute_signals = ['pupil']
    elif loadkey.endswith("pupbeh0"):
        state_signals = ['pupil', 'active']
        permute_signals = ['active']
    elif loadkey.endswith("pupbeh"):
        state_signals = ['pupil', 'active']
        permute_signals = []

    elif loadkey.endswith("pup0pre0beh"):
        state_signals = ['pupil', 'pre_passive', 'active']
        permute_signals = ['pupil', 'pre_passive']
    elif loadkey.endswith("puppre0beh"):
        state_signals = ['pupil', 'pre_passive', 'active']
        permute_signals = ['pre_passive']
    elif loadkey.endswith("pup0prebeh"):
        state_signals = ['pupil', 'pre_passive', 'active']
        permute_signals = ['pupil']
    elif loadkey.endswith("pupprebeh"):
        state_signals = ['pupil', 'pre_passive', 'active']
        permute_signals = []

    elif loadkey.endswith("pre0beh0"):
        state_signals = ['pre_passive', 'active']
        permute_signals = ['pre_passive', 'active']
    elif loadkey.endswith("pre0beh"):
        state_signals = ['pre_passive', 'active']
        permute_signals = ['pre_passive']
    elif loadkey.endswith("prebeh0"):
        state_signals = ['pre_passive', 'active']
        permute_signals = ['active']
    elif loadkey.endswith("prebeh"):
        state_signals = ['pre_passive', 'active']
        permute_signals = []

    elif loadkey.endswith("predif0beh"):
        state_signals = ['pre_passive', 'puretone_trials',
                         'hard_trials', 'active']
        permute_signals = ['puretone_trials', 'hard_trials']
    elif loadkey.endswith("predifbeh"):
        state_signals = ['pre_passive', 'puretone_trials',
                         'hard_trials', 'active']
        permute_signals = []
    elif loadkey.endswith("pbs0pev0beh0"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['pupil_bs', 'pupil_ev', 'active']
    elif loadkey.endswith("pbspev0beh"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['pupil_ev']
    elif loadkey.endswith("pbs0pevbeh"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['pupil_bs']
    elif loadkey.endswith("pbspevbeh0"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['pupil_bs', 'pupil_ev']
    elif loadkey.endswith("pbs0pev0beh"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['active']
    elif loadkey.endswith("pbspevbeh"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = []
    else:
        raise ValueError("invalid loadkey string")


def env(loadkey, recording_uri):

    recordings = [recording_uri]
    state_signals, permute_signals = _state_model_loadkey_helper(loadkey)
    xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings}],
                  ['nems.xforms.make_state_signal',
                   {'state_signals': state_signals,
                    'permute_signals': permute_signals,
                    'new_signalname': 'state'}]]
    return xfspec


def psth(loadkey, recording_uri):

    recordings = [recording_uri]
    state_signals, permute_signals = _state_model_loadkey_helper(loadkey)
    xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings}],
                  ['nems.xforms.generate_psth_from_resp', {}],
                  ['nems.xforms.make_state_signal',
                   {'state_signals': state_signals,
                    'permute_signals': permute_signals,
                    'new_signalname': 'state'}]]
    return xfspec


def psths(loadkey, recording_uri):

    recordings = [recording_uri]
    state_signals, permute_signals = _state_model_loadkey_helper(loadkey)
    xfspec = [['nems.xforms.load_recordings',
               {'recording_uri_list': recordings}],
              ['nems.xforms.generate_psth_from_resp',
               {'smooth_resp': True}],
              ['nems.xforms.make_state_signal',
               {'state_signals': state_signals,
                'permute_signals': permute_signals,
                'new_signalname': 'state'}]]
    return xfspec


loader_lib = {
        'ozgf': ozgf,
        'env': env,
        'psth': psth,
        'psths': psths
        }



def default_fitter_xfspec(fitkey, kwargs):
    return None


def fit_model_xforms(recording_uri, modelname, fitkey_kwargs=None,
                     autoPlot=True):
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

    loader_fn = getattr(imp.import_module(get_setting('XF_LOADER_MODULE')),
                        get_setting('XF_LOADER_FN'))
    loader_xfspec = loader_fn(loadkey, recording_uri)
    # If None, don't use a separate fitter key - apply default
    # loader xfspec instead.
    if loader_xfspec is not None:
        kws.pop(0)
    else:
        loader_xfspec = [['nems.xforms.load_recordings',
                          {'recording_uri_list': [recording_uri]}]]

    fitter_fn = getattr(imp.import_module(get_setting('XF_FITTER_MODULE')),
                        get_setting('XF_FITTER_FN'))
    fitter_xfspec = fitter_fn(fitkey, fitkey_kwargs)
    # If None, don't use a separate fitter key - apply default
    # fitter xfspec instead.
    if fitter_xfspec is not None:
        kws.pop(-1)
    else:
        fitter_xfspec = [['nems.xforms.fit_basic_init', {}],
                         ['nems.forms.fit_basic', {}],
                         ['nems.xforms.predict', {}]]

    modelspecname = "_".join(kws)

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
