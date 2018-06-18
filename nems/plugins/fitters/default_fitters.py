import logging
import re

from nems.fitters.fitter import coordinate_descent, scipy_minimize

log = logging.getLogger(__name__)

# TODO: Create and expand documentation.
# TODO: set up alias'ing function similar to one for loaders.


def basic(fitkey):
    '''
    TODO: expand this doc

    Format:
        basic.<fitter>.<jackknifing>.<metric>.<misc>
    Example string:
        basic.cd.nf10.shr.mi1000
    Translation:
        Use fit_basic with coordinate_descent, nfold fitting with 10 folds,
        with nmse_shrinkage as the metric and 1000 maximum iterations
    '''

    xfspec = []
    if fitkey == 'basic':
        # set up reasonable defaults
        pass  # TODO
    else:
        options = _extract_options(fitkey)
        metric, nfold, fitter, state = _parse_fit(options)
        max_iter = _parse_basic(options)
        if nfold:
            log.info("n-fold fitting...")
            xfspec.append(['nems.xforms.mask_for_jackknife',
                           {'njacks': nfold, 'epoch_name': 'REFERENCE'}])
            if state:
                xfspec.append(['nems.xforms.fit_state_init',
                               {'metric': metric}])
            xfspec.extend([['nems.xforms.fit_nfold',
                            {'fitter': fitter, 'metric': metric}]
                           ['nems.xforms.predict', {}]])
        else:
            if state:
                xfspec.append(['nems.xforms.fit_state_init',
                               {'metric': metric}])
            else:
                xfspec.append(['nems.xforms.fit_basic_init',
                               {'metric': metric}])
            xfspec.extend([['nems.xforms.fit_basic',
                            {'metric': metric, 'max_iter': max_iter,
                             'fitter': fitter}],
                           ['nems.xforms.predict', {}]])

    return xfspec


def iter(fitkey):
    if fitkey == 'iter':
        # set up reasonable defaults
        xfspec = []  # TODO
    else:
        # TODO: Support nfold and state fits for fit_iteratively?
        metric, nfold, fitter, state = _parse_fit(fitkey)
        tolerances, module_sets, fit_iter, tol_iter = _parse_iter(fitkey)

        xfspec = [['nems.xforms.fit_basic_init', {'tolerance': 1e-4}],
                  ['nems.xforms.fit_iteratively',
                   {'module_sets': module_sets, 'fitter': fitter,
                    'tolerances': tolerances, 'tol_iter': tol_iter,
                    'fit_iter': fit_iter, 'metric': metric}],
                  ['nems.xforms.predict', {}]]

    return xfspec


def _extract_options(fitkey):
    chunks = fitkey.split('.')
    options = chunks[1:]
    return options


def _parse_fit(options):
    '''For general fitting options that apply to most all fitters.'''
    # declare default settings
    metric = 'nmse'
    nfold = 0
    fitter = scipy_minimize
    state = False

    # override defaults where appropriate
    for op, i in enumerate(options):
        # check for shrinkage
        if op == 'shr':
            metric = 'nmse_shrink'
        elif 'nf' in op:
            nf = re.compile(r'^nf{\d{0,}$')
            nfold = int(re.match(nf, op)[1])
        elif op == 'cd':
            fitter = coordinate_descent
        elif op == 's':
            state = True

    return metric, nfold, fitter, state


def _parse_basic(options):
    '''Options specific to basic.'''
    max_iter = 1000
    for op in options:
        if op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            max_iter = int(re.match(pattern, op)[1])
        else:
            pass  # TODO

    return max_iter


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
