import logging

from nems.fitters.api import coordinate_descent, scipy_minimize
import nems.xforms as xforms

log = logging.getLogger(__name__)


def generate_loader_xfspec(loader, recording_uri):

    recordings = [recording_uri]

    if loader in ["ozgf100ch18", "ozgf100ch18n"]:
        normalize = int(loader == "ozgf100ch18n")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.split_by_occurrence_counts',
                   {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences',{}]]

    elif loader in ["ozgf100ch18pup", "ozgf100ch18npup"]:
        normalize = int(loader == "ozgf100ch18npup")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.make_state_signal',
                   {'state_signals': ['pupil'], 'permute_signals': [],
                    'new_signalname': 'state'}]]

    elif loader == "nostim10pup":
        # DEPRECATED?
        xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.preprocessing.make_state_signal', {'state_signals': ['pupil'], 'permute_signals': [], 'new_signalname': 'state'},['rec'],['rec']]]

    elif loader in ["nostim10pup0beh0","nostim10pup0beh",
                    "nostim10pupbeh0","nostim10pupbeh",
                    "nostim20pup0beh0", "nostim20pup0beh",
                    "nostim20pupbeh0", "nostim20pupbeh"]:

        state_signals = ['pupil', 'active']

        if loader.endswith("pup0beh0"):
            permute_signals = ['pupil', 'active']
        elif loader.endswith("pup0beh"):
            permute_signals = ['pupil']
        elif loader.endswith("pupbeh0"):
            permute_signals = ['active']
        elif loader.endswith("pupbeh"):
            permute_signals = []
        else:
            raise ValueError("invalid loader string")

        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings}],
                  ['nems.xforms.make_state_signal',
                   {'state_signals': state_signals,
                    'permute_signals': permute_signals,
                    'new_signalname': 'state'}]]

    elif loader in ["env100","env100n"]:
        normalize = int(loader == "env100n")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.split_by_occurrence_counts',
                   {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences', {}]]

    else:
        raise ValueError('unknown loader string')

    return xfspec


def generate_fitter_xfspec(fitkey, fitkey_kwargs=None):

    xfspec = []

    # parse the fit spec: Use gradient descent on whole data set(Fast)
    if fitkey in ["fit01", "basic"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey in ["fit01a", "basicqk"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic',
                       {'max_iter': 1000, 'tolerance': 1e-5}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey in ["fit01b", "basic-shr"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic',
                       {'shrinkage': 1, 'tolerance': 1e-8}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey in ["fit01b", "basic-cd"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic_cd', {'shrinkage': 0}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey in ["fit01b", "basic-cd-shr"]:
        # prefit strf
        xfspec.append(['nems.xforms.fit_basic_init', {}])
        xfspec.append(['nems.xforms.fit_basic_cd',
                       {'shrinkage': 1, 'tolerance': 1e-8}])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey == "fitjk01":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif (fitkey == "fitpjk01") or (fitkey == "basic-nf"):

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': 10, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_nfold', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "basic-nf-shr":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': 10, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_nfold_shrinkage', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "cd-nf-shr":

        log.info("n-fold fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': 10, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_cd_nfold_shrinkage', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "iter-cd-nf-shr":

        log.info("Iterative cd, n-fold, shrinkage fitting...")
        xfspec.append(['nems.xforms.split_for_jackknife',
                       {'njacks': 10, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems.xforms.fit_iter_cd_nfold_shrink', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "fit02":
        # no pre-fit
        log.info("Performing full fit...")
        xfspec.append(['nems.xforms.fit_basic', {}])
        xfspec.append(['nems.xforms.predict',    {}])

    elif fitkey == "fitsubs":
        '''fit_subsets with scipy_minimize'''
        kw_list = ['module_sets', 'tolerance', 'fitter']
        defaults = [None, 1e-4, coordinate_descent]
        module_sets, tolerance, my_fitter = \
            _get_my_kwargs(fitkey_kwargs, kw_list, defaults)
        xfspec.append([
                'nems.xforms.fit_module_sets',
                {'module_sets': module_sets, 'fitter': scipy_minimize,
                 'tolerance': tolerance}
                ])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey.startswith("fitsubs"):
        xfspec.append(_parse_fitsubs(fitkey))
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey == "fititer":
        kw_list = ['module_sets', 'tolerances', 'tol_iter', 'fit_iter',
                   'fitter']
        defaults = [None, None, 100, 20, coordinate_descent]
        module_sets, tolerances, tol_iter, fit_iter, my_fitter = \
            _get_my_kwargs(fitkey_kwargs, kw_list, defaults)
        xfspec.append([
                'nems.xforms.fit_iteratively',
                {'module_sets': module_sets, 'fitter': my_fitter,
                 'tolerances': tolerances, 'tol_iter': tol_iter,
                 'fit_iter': fit_iter}
                ])
        xfspec.append(['nems.xforms.predict', {}])

    elif fitkey.startswith("fititer"):
        xfspec.append(_parse_fititer(fitkey))
        xfspec.append(['nems.xforms.predict', {}])

    else:
        raise ValueError('unknown fitter string ' + fitkey)

    return xfspec


def _get_my_kwargs(kwargs, kw_list, defaults):
    '''Fetch value of kwarg if given, otherwise corresponding default'''
    my_kwargs = []
    for kw, default in zip(kw_list, defaults):
        if kwargs is None:
            a = default
        else:
            a = kwargs.pop(kw, default)
        my_kwargs.append(a)
    return my_kwargs


def _parse_fititer(fit_keyword):
    # ex: fititer01-T4-T6-S0x1-S0x1x2x3-ti50-fi20
    # fitter: scipy_minimize; tolerances: [1e-4, 1e-6]; s
    # subsets: [[0,1], [0,1,2,3]]; tol_iter: 50; fit_iter: 20;
    # Note that order does not matter except for starting with
    # 'fititer<some number>' to specify the analysis and fit algorithm
    chunks = fit_keyword.split('-')

    fit = chunks[0]
    if fit.endswith('01'):
        fitter = scipy_minimize
    elif fit.endswith('02'):
        fitter = coordinate_descent
    else:
        fitter = coordinate_descent
        log.warn("Unrecognized or unspecified fit algorithm for fititer: %s\n"
                 "Using default instead: %s", fit[7:], fitter)

    tolerances = []
    module_sets = []
    fit_iter = None
    tol_iter = None

    for c in chunks[1:]:
        if c.startswith('ti'):
            tol_iter = int(c[2:])
        elif c.startswith('fi'):
            fit_iter = int(c[2:])
        elif c.startswith('T'):
            power = int(c[1:])*-1
            tol = 10**(power)
            tolerances.append(tol)
        elif c.startswith('S'):
            indices = [int(i) for i in c[1:].split('x')]
            module_sets.append(indices)
        else:
            log.warning(
                    "Unrecognized segment in fititer keyword: %s\n"
                    "Correct syntax is:\n"
                    "fititer<fitter>-S<i>x<j>...-T<tolpower>...ti<tol_iter>"
                    "-fi<fit_iter>", c
                    )


    if not tolerances:
        tolerances = None
    if not module_sets:
        module_sets = None

    return ['nems.xforms.fit_iteratively',
            {'module_sets': module_sets, 'fitter': fitter,
             'tolerances': tolerances, 'tol_iter': tol_iter,
             'fit_iter': fit_iter}]


def _parse_fitsubs(fit_keyword):
    # ex: fitsubs02-S0x1-S0x1x2x3-it1000-T6
    # fitter: scipy_minimize; subsets: [[0,1], [0,1,2,3]];
    # max_iter: 1000;
    # Note that order does not matter except for starting with
    # 'fitsubs<some number>' to specify the analysis and fit algorithm
    chunks = fit_keyword.split('-')

    fit = chunks[0]
    if fit.endswith('01'):
        fitter = scipy_minimize
    elif fit.endswith('02'):
        fitter = coordinate_descent
    else:
        fitter = coordinate_descent
        log.warn("Unrecognized or unspecified fit algorithm for fitsubs: %s\n"
                 "Using default instead: %s", fit[7:], fitter)

    module_sets = []
    max_iter = None
    tolerance = None

    for c in chunks[1:]:
        if c.startswith('it'):
            max_iter = int(c[2:])
        elif c.startswith('S'):
            indices = [int(i) for i in c[1:].split('x')]
            module_sets.append(indices)
        elif c.startswith('T'):
            power = int(c[1:])*-1
            tolerance = 10**(power)
        else:
            log.warning(
                    "Unrecognized segment in fitsubs keyword: %s\n"
                    "Correct syntax is:\n"
                    "fitsubs<fitter>-S<i>x<j>...-T<tolpower>-it<max_iter>", c
                    )

    if not module_sets:
        module_sets = None

    return ['nems.xforms.fit_iteratively',
            {'module_sets': module_sets, 'fitter': fitter,
             'tolerance': tolerance, 'max_iter': max_iter}]

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
    fitkey = kws[-1]

    meta = {'modelname': modelname, 'loader': loader, 'fitkey': fitkey,
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
    xfspec += generate_fitter_xfspec(fitkey, fitkey_kwargs)

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
