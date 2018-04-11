import logging

from nems.fitters.api import coordinate_descent, scipy_minimize
import nems.xforms as xforms

log = logging.getLogger(__name__)


def generate_loader_xfspec(loader, recording_uri):

    recordings = [recording_uri]

    options = {}
    if loader == "ozgf100ch18":
        options["stimfmt"] = "ozgf"
        options["chancount"] = 18
        options["rasterfs"] = 100
        options['includeprestim'] = 1
        options["average_stim"]=True
        options["state_vars"]=[]
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences',{}]]

    elif loader == "ozgf100ch18pup":
        options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf',
          'chancount': 18, 'pupil': True, 'stim': True,
          'pupil_deblink': True, 'pupil_median': 1}
        options["average_stim"]=False
        options["state_vars"]=['pupil']
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.xforms.make_state_signal', {'state_signals': ['pupil'], 'permute_signals': [], 'new_signalname': 'state'}]]

    elif loader == "nostim10pup":
        options={'rasterfs': 10, 'includeprestim': True, 'stimfmt': 'parm',
          'chancount': 0, 'pupil': True, 'stim': False,
          'pupil_deblink': True, 'pupil_median': 1}
        options["average_stim"]=False
        options["state_vars"]=['pupil']
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.preprocessing.make_state_signal', {'state_signals': ['pupil'], 'permute_signals': [], 'new_signalname': 'state'},['rec'],['rec']]]

    elif loader in ["nostim10pup0beh0","nostim10pup0beh","nostim10pupbeh0","nostim10pupbeh"]:
        options={'rasterfs': 10, 'includeprestim': True, 'stimfmt': 'parm',
          'chancount': 0, 'pupil': True, 'stim': False,
          'pupil_deblink': True, 'pupil_median': 1}
        options["average_stim"]=False
        options["state_vars"]=['pupil']

        state_signals=['pupil','behavior_state']
        if loader=="nostim10pup0beh0":
            permute_signals=['pupil','behavior_state']
        elif loader=="nostim10pup0beh":
            permute_signals=['pupil']
        elif loader=="nostim10pupbeh0":
            permute_signals=['behavior_state']
        else:
            permute_signals=['']

        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings}],
                  ['nems.xforms.make_state_signal',
                   {'state_signals': state_signals,
                    'permute_signals': permute_signals,
                    'new_signalname': 'state'}]]

    elif loader in ["nostim20pup0beh0", "nostim20pup0beh",
                    "nostim20pupbeh0", "nostim20pupbeh"]:
        options = {'rasterfs': 20, 'includeprestim': True, 'stimfmt': 'parm',
                   'chancount': 0, 'pupil': True, 'stim': False,
                   'pupil_deblink': True, 'pupil_median': 1}
        options["average_stim"] = False
        options["state_vars"] = ['pupil']

        state_signals=['pupil','active']
        if loader=="nostim20pup0beh0":
            permute_signals=['pupil','active']
        elif loader=="nostim20pup0beh":
            permute_signals=['pupil']
        elif loader=="nostim20pupbeh0":
            permute_signals=['active']
        else:
            permute_signals=['']

        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.preprocessing.make_state_signal', {'state_signals': state_signals, 'permute_signals': permute_signals, 'new_signalname': 'state'},['rec'],['rec']]]

    elif loader == "env100":
        options["stimfmt"] = "envelope"
        options["chancount"] = 0
        options["rasterfs"] = 100
        options['includeprestim'] = 1
        options["average_stim"]=True
        options["state_vars"]=[]
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences',{}]]

    else:
        raise ValueError('unknown loader string')

    return xfspec


def generate_fitter_xfspec(fitter, fitter_kwargs=None):

    xfspec=[]

    # parse the fit spec: Use gradient descent on whole data set(Fast)
    if fitter == "fit01":
        # prefit strf
        log.info("Prefitting STRF without other modules...")
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitter == "fit01a":
        # prefit strf
        log.info("Prefitting STRF without other modules...")
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic',
                       {'maxiter': 1000, 'ftol': 1e-5}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitter == "fit01b":
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic_shrink', {}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitter == "fitjk01":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife', {'njacks': 5}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitter == "fitpjk01":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife', {'njacks': 10}])
        xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitter == "fit02":
        # no pre-fit
        log.info("Performing full fit...")
        xfspec.append(['nems.xforms.fit_basic', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitter == "fititer01":
        '''fit_iteratively with scipy_minimize'''
        kw_list = ['module_sets', 'tolerances', 'invert', 'max_iter']
        defaults = [None, None, False, 100]
        module_sets, tolerances, invert, max_iter = \
            _get_my_kwargs(fitter_kwargs, kw_list, defaults)
        xfspec.append([
                'nems.xforms.fit_iteratively',
                {'module_sets':module_sets, 'tolerances': tolerances,
                 'invert': invert, 'max_iter': max_iter,
                 'fitter': scipy_minimize}
                ])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitter == "fititer02":
        '''fit_iteratively with coordinate_descent'''
        kw_list = ['module_sets', 'tolerances', 'invert', 'max_iter']
        defaults = [None, None, False, 100]
        module_sets, tolerances, invert, max_iter = \
            _get_my_kwargs(fitter_kwargs, kw_list, defaults)
        xfspec.append([
                'nems.xforms.fit_iteratively',
                {'module_sets':module_sets, 'tolerances': tolerances,
                 'invert': invert, 'max_iter': max_iter,
                 'fitter': coordinate_descent}
                ])
        xfspec.append(['nems.xforms.predict', {}])

    else:
        raise ValueError('unknown fitter string ' + fitter)

    return xfspec


def _get_my_kwargs(kwargs, kw_list, defaults):
    '''Fetch value of kwarg if given, otherwise corresponding default'''
    my_kwargs = []
    for i, kw in enumerate(kw_list):
        if kwargs is None:
            a = defaults[i]
        else:
            a = kwargs.pop(kw, defaults[i])
        my_kwargs.append(a)
    return my_kwargs


# TODO: take baphy out of names and docs if we're keeping these here.
#       (fit and load)
# TODO: Need loader_kwargs for anything, similar to fitter_kwargs?
#       leaving out for now but could be useful.
def fit_model_xforms(recording_uri, modelname, fitter_kwargs=None,
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

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitter = kws[-1]

    meta = {'modelname': modelname, 'loader': loader, 'fitter': fitter,
            'modelspecname': modelspecname}

    # TODO: These should be added to meta by nems_db after ctx is returned.
    #       'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
    #       'githash': os.environ.get('CODEHASH', ''),
    #       'recording': loader}

    # Generate the xfspec, which defines the sequence of events
    # to run through (like a packaged-up script)

    # 1) Load the data
    xfspec = generate_loader_xfspec(loader, recording_uri)

    # 2) generate a modelspec
    xfspec.append(['nems.xforms.init_from_keywords',
                   {'keywordstring': modelspecname, 'meta': meta}])

    # 3) fit the data
    xfspec += generate_fitter_xfspec(fitter, fitter_kwargs)

    # 4) add some performance statistics
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs'], ['modelspecs']])

    # 5) generate plots
    if autoPlot:
        log.info('Generating summary plot...')
        xfspec.append(['nems.xforms.plot_summary', {}])

    # Now that the xfspec is assembled, run through it
    # in order to get the fitted modelspec, evaluated recording, etc.
    # (all packaged up in the ctx dictionary).
    ctx, log_xf = xforms.evaluate(xfspec)

    return ctx
