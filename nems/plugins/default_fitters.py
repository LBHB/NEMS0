import logging
import re

log = logging.getLogger(__name__)

# TODO: Create and expand documentation.
# TODO: set up alias'ing function similar to one for loaders.


def basic(fitkey):
    '''
    Perform a fit_basic analysis on a model.

    Parameters
    ----------
    fitkey : str
        Expected format: basic.<fitter>.<jackknifing>.<metric>.<misc>
        Example: basic.cd.nf10.shr.mi1000
        Example translation:
            Use fit_basic with coordinate_descent, nfold fitting with 10 folds,
            with nmse_shrinkage as the metric and 1000 maximum iterations

    Options
    -------
    cd : Use coordinate_descent for fitting (default is scipy_minimize)
    shr : Use nmse_shrink as the performance metric (default is nmse)
    nfN : Perform nfold fitting with N folds, where N is any positive integer.
    st : Expect a model based on state variables.
    miN : Set maximum iterations to N, where N is any positive integer.
    tN : Set tolerance to 10**-N, where N is any positive integer.
    npr : Skip the default prefitting routine for non-state models.
    m : Use split_for_jackknife instead of mask_for_jackknife

    '''

    xfspec = []

    options = _extract_options(fitkey)
    metric, nfold, fitter, state, epoch = _parse_fit(options)
    max_iter, tolerance, prefit, do_split = _parse_basic(options)
    if nfold:
        log.info("n-fold fitting...")
        jk_kwargs = {'metric': metric, 'tolerance': tolerance}
        if isinstance(nfold, int):
            # if nfold is just True instead of an integer, use
            # default number specified in xforms function
            jk_kwargs['njacks'] = nfold

        if do_split:
            xfspec.append(['nems.xforms.split_for_jackknife', jk_kwargs])
        else:
            xfspec.append(['nems.xforms.mask_for_jackknife', jk_kwargs])

        if state:
            xfspec.append(['nems.xforms.fit_state_init',
                           {'metric': metric}])
        xfspec.extend([['nems.xforms.fit_nfold',
                        {'fitter': fitter, 'metric': metric,
                         'tolerance': tolerance}]])
    else:
        if state:
            xfspec.append(['nems.xforms.fit_state_init',
                           {'metric': metric}])
        else:
            if prefit:
                xfspec.append(['nems.xforms.fit_basic_init',
                               {'metric': metric}])
        xfspec.extend([['nems.xforms.fit_basic',
                        {'metric': metric, 'max_iter': max_iter,
                         'fitter': fitter, 'tolerance': tolerance}]])

    # Always have predict at end regardless of options
    xfspec.append(['nems.xforms.predict', {}])

    return xfspec


def iter(fitkey):
    '''
    Perform a fit_iteratively analysis on a model.

    Parameters
    ----------
    fitkey : str
        Expected format: iter.<fitter>.<jackknifing>.<metric>.<misc>
        Example: iter.nf5.T3,4,7.S0,1.S2,3.S0,1,2,3.ti50.fi20
        Example translation:
            Use fit_iteratively with scipy_minimize (since 'cd' option is not
            present), nfold fitting with 5 folds, nmse as the metric
            (since 'shr' option is not present), 50 per-tolerance-level
            iterations, and 20 per-fit iterations.
            Begin with a tolerance level of 10**-3, followed by
            10**-4 and 10**-7. Within each tolerance level,
            first fit modules 0 and 1, then 2 and 3,
            and finally 0, 1, 2, and 3 all together.

    Options
    -------
    cd : Use coordinate_descent for fitting (default is scipy_minimize)
    shr : Use nmse_shrink as the performance metric (default is nmse)
    nfN : Perform nfold fitting with N folds, where N is any positive integer.
    st : Expect a model based on state variables.
    TN,N,... : Use tolerance levels 10**-N for each N given, where N is
               any positive integer.
    SN,N,... : Fit model indices N, N... for each N given,
               where N is any positive integer or zero. May be provided
               multiple times to iterate over several successive subsets.
    tiN : Perform N per-tolerance-level iterations, where N is any
          positive integer.
    fiN : Perform N per-fit iterations, where N is any positive integer.

    Note
    ----
    Nfold and state fits are not currently supported for this
    analysis type, WIP.
    '''

    # TODO: Support nfold and state fits for fit_iteratively?
    #       And epoch to go with state.
    options = _extract_options(fitkey)
    metric, nfold, fitter, state, epoch = _parse_fit(options)
    tolerances, module_sets, fit_iter, tol_iter = _parse_iter(options)

    xfspec = [['nems.xforms.fit_basic_init', {'tolerance': 1e-4}],
              ['nems.xforms.fit_iteratively',
               {'module_sets': module_sets, 'fitter': fitter,
                'tolerances': tolerances, 'tol_iter': tol_iter,
                'fit_iter': fit_iter, 'metric': metric}],
              ['nems.xforms.predict', {}]]

    return xfspec


def _extract_options(fitkey):
    if fitkey == 'basic' or fitkey == 'iter':
        # empty options (i.e. just use defualts)
        options = []
    else:
        chunks = fitkey.split('.')
        options = chunks[1:]
    return options


def _parse_fit(options):
    '''For general fitting options that apply to most all fitters.'''
    # declare default settings
    metric = 'nmse'
    nfold = 0
    fitter = 'scipy_minimize'
    state = False
    epoch = 'REFERENCE'
    # TODO: Find a better solution for default epoch - REFERENCE is
    #       lbhb-specific, but don't want to have to put epREFERENCE in every
    #       model name.

    # override defaults where appropriate
    for op in options:
        # check for shrinkage
        if op == 'shr':
            metric = 'nmse_shrink'
        elif op.startswith('nf'):
            pattern = re.compile(r'^nf(\d{0,})$')
            nfold = re.match(pattern, op).group(1)
            if nfold:
                nfold = int(nfold)
            else:
                nfold = True  # Use nfold but defer to default number
        elif op == 'cd':
            fitter = 'coordinate_descent'
        elif op == 'st':
            state = True
        elif op.startswith('ep'):
            pattern = re.compile(r'^ep(\w{1,})$')
            epoch = re.match(pattern, op).group(1)

    return metric, nfold, fitter, state, epoch


def _parse_basic(options):
    '''Options specific to basic.'''
    max_iter = 1000
    tolerance = 1e-7
    prefit = True  # TODO: Still need this option, or always want prefit?
    do_split = False
    for op in options:
        if op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            max_iter = int(re.match(pattern, op).group(1))
        elif op.startswith('t'):
            pattern = re.compile(r'^t(\d{1,})')
            power = int(re.match(pattern, op).group(1))*(-1)
            tolerance = 10**power
        elif op == 'npr':
            prefit = False
        elif op == 'm':
            do_split = True

    return max_iter, tolerance, prefit, do_split


def _parse_iter(options):
    '''Options specific to iter.'''
    tolerances = []
    module_sets = []
    fit_iter = None
    tol_iter = None

    for op in options:
        if op.startswith('ti'):
            tol_iter = int(op[2:])
        elif op.startswith('fi'):
            fit_iter = int(op[2:])
        elif op.startswith('T'):
            powers = [int(i) for i in op[1:].split(',')]
            tolerances.extend([10**(-1*p) for p in powers])
        elif op.startswith('S'):
            indices = [int(i) for i in op[1:].split(',')]
            module_sets.append(indices)
        else:
            pass

    if not tolerances:
        tolerances = None
    if not module_sets:
        module_sets = None

    return tolerances, module_sets, fit_iter, tol_iter


# TODO: Redo this for new setup after deciding if still needed
#       (might only need _parse_iter)


def _parse_fitsubs(fit_keyword):
    # ex: fitsubs02-S0x1-S0x1x2x3-it1000-T6
    # fitter: scipy_minimize; subsets: [[0,1], [0,1,2,3]];
    # max_iter: 1000;
    # Note that order does not matter except for starting with
    # 'fitsubs<some number>' to specify the analysis and fit algorithm
    chunks = fit_keyword.split('-')

    fit = chunks[0]
    if fit.endswith('01'):
        fitter = 'scipy_minimize'
    elif fit.endswith('02'):
        fitter = 'coordinate_descent'
    else:
        fitter = 'coordinate_descent'
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
