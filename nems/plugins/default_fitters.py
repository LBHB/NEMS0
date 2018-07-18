import logging
import re

log = logging.getLogger(__name__)

# TODO: Create and expand documentation.
# TODO: set up alias'ing function similar to one for loaders.


def mt(fitkey):
    '''
    Set fitting metric to nmse_shrinkage for keywords that come after.

    TODO: Any other metrics as added options?

    '''
    ops = fitkey.split('.')[1:]
    if 'shr' in ops:
        metric = 'nmse_shrink'
    else:
        # Redundant for now since shr is the only option, but will likely
        # add more later.
        metric = 'nmse_shrink'
    return [['nems.xforms.use_metric', {'metric': metric}]]


def pred(fitkey):
    '''
    Evaluate model prediction. Added by default in xform_helper.
    '''
    return [['nems.xforms.predict', {}]]


def stats(fitkey):
    '''
    Add summary statistics to modelspec(s). Added by default in xform_helper.
    '''
    options = fitkey.split('.')[1:]
    fn = 'standard_correlation'
    for op in options:
        if op == 'pm':
            fn = 'correlation_per_model'

    return [['nems.xforms.add_summary_statistics', {'fn': fn}]]


def best(fitkey):
    '''
    Collapse modelspecs to singleton list with only the "best" modelspec.
    '''
    options = fitkey.split('.')[1:]
    metakey = 'r_test'

    # TODO: need to update syntax so that a separate option for every
    #       meta field isn't necessary (currently can't handle underscores)
    #       Would be easy to parse here but underscore would still mess up the
    #       modelname split in xform_helper
    for op in options:
        if op == 'rtest':
            fitkey = 'r_test'
        elif op == 'rfit':
            fitkey = 'r_fit'

    return [['nems.xforms.only_best_modelspec', {'metakey': metakey}]]


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
    miN : Set maximum iterations to N, where N is any positive integer.
    tN : Set tolerance to 10**-N, where N is any positive integer.

    '''

    xfspec = []

    options = _extract_options(fitkey)
    max_iter, tolerance, fitter = _parse_basic(options)
    xfspec = [['nems.xforms.fit_basic',
               {'max_iter': max_iter,
                'fitter': fitter, 'tolerance': tolerance}]]

    return xfspec


def iter(fitkey):
    '''
    Perform a fit_iteratively analysis on a model.

    Parameters
    ----------
    fitkey : str
        Expected format: iter.<fitter>.<misc>
        Example: iter.T3,4,7.S0,1.S2,3.S0,1,2,3.ti50.fi20
        Example translation:
            Use fit_iteratively with scipy_minimize
            (since 'cd' option is not present), 50 per-tolerance-level
            iterations, and 20 per-fit iterations.
            Begin with a tolerance level of 10**-3, followed by
            10**-4 and 10**-7. Within each tolerance level,
            first fit modules 0 and 1, then 2 and 3,
            and finally 0, 1, 2, and 3 all together.

    Options
    -------
    cd : Use coordinate_descent for fitting (default is scipy_minimize)
    TN,N,... : Use tolerance levels 10**-N for each N given, where N is
               any positive integer.
    SN,N,... : Fit model indices N, N... for each N given,
               where N is any positive integer or zero. May be provided
               multiple times to iterate over several successive subsets.
    tiN : Perform N per-tolerance-level iterations, where N is any
          positive integer.
    fiN : Perform N per-fit iterations, where N is any positive integer.

    '''

    # TODO: Support nfold and state fits for fit_iteratively?
    #       And epoch to go with state.
    options = _extract_options(fitkey)
    tolerances, module_sets, fit_iter, tol_iter, fitter = _parse_iter(options)

    xfspec = [['nems.xforms.fit_iteratively',
               {'module_sets': module_sets, 'fitter': fitter,
                'tolerances': tolerances, 'tol_iter': tol_iter,
                'fit_iter': fit_iter}]]

    return xfspec


def _extract_options(fitkey):
    if fitkey == 'basic' or fitkey == 'iter':
        # empty options (i.e. just use defualts)
        options = []
    else:
        chunks = fitkey.split('.')
        options = chunks[1:]
    return options


def _parse_basic(options):
    '''Options specific to basic.'''
    max_iter = 1000
    tolerance = 1e-7
    fitter = 'scipy_minimize'
    for op in options:
        if op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            max_iter = int(re.match(pattern, op).group(1))
        elif op.startswith('t'):
            num = op.replace('d', '.')
            tolpower = float(num[1:])*(-1)
            tolerance = 10**tolpower
        elif op == 'cd':
            fitter = 'coordinate_descent'

    return max_iter, tolerance, fitter


def _parse_iter(options):
    '''Options specific to iter.'''
    tolerances = []
    module_sets = []
    fit_iter = 10
    tol_iter = 50
    fitter = 'scipy_minimize'

    for op in options:
        if op.startswith('ti'):
            tol_iter = int(op[2:])
        elif op.startswith('fi'):
            fit_iter = int(op[2:])
        elif op.startswith('T'):
            nums = op.replace('d', '.')
            powers = [float(i) for i in nums[1:].split(',')]
            tolerances.extend([10**(-1*p) for p in powers])
        elif op.startswith('S'):
            indices = [int(i) for i in op[1:].split(',')]
            module_sets.append(indices)
        elif op == 'cd':
            fitter = 'coordinate_descent'

    if not tolerances:
        tolerances = None
    if not module_sets:
        module_sets = None

    return tolerances, module_sets, fit_iter, tol_iter, fitter
