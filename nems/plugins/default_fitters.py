import logging
import re

from nems.utils import escaped_split, keyword_extract_options

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
    options = escaped_split(fitkey, '.')[1:]
    metakey = 'r_test'
    comparison = 'greatest'

    for op in options:
        if op == '<':
            comparison = 'least'
        elif op == '>':
            comparison = 'greatest'
        else:
            # Assume it's the name of a metakey, and remove any escapes
            metakey = op.replace('\\', '')

    return [['nems.xforms.only_best_modelspec', {'metakey': metakey,
                                                 'comparison': comparison}]]


def sort(fitkey):
    '''
    Sorts modelspecs by specified meta entry in either descending or
    ascending order.
    '''
    ops = escaped_split(fitkey, '.')[1:]
    metakey = 'r_test'
    order = 'descending'

    for op in ops:
        if op in ['a', 'asc', 'ascending']:
            order = op
        elif op in ['d', 'desc', 'descending']:
            order = op
        else:
            # Assume it's the name of a metakey, and remove any escapes
            metakey = op.replace('\\', '')

    return [['nems.xforms.sort_modelspecs', {'metakey': metakey,
                                             'order': order}]]


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
    max_iter, tolerance, fitter, choose_best, fast_eval, rand_count = _parse_basic(options)
    xfspec = []
    #if fast_eval:
    #    xfspec = [['nems.xforms.fast_eval', {}]]
    #else:
    #    xfspec = []
    if rand_count>1:
        xfspec.append(['nems.initializers.rand_phi', {'rand_count': rand_count}])
    xfspec.append(['nems.xforms.fit_basic',
                  {'max_iter': max_iter,
                   'fitter': fitter, 'tolerance': tolerance}])
    if choose_best:
        xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

    return xfspec


def nrc(fitkey):
    '''
    Use normalized reverse correlation to fit phi. Right now, pretty dumb.
    Expects two modules (fir and lvl). Will fit both simultaneously. Can stack
    this with fit basic in order to initialize a full rank model with the reverse
    correlation solution.
    '''
    xfspec = []

    #options = _extract_options(fitkey)
    #max_iter, tolerance, fitter = _parse_basic(options)
    xfspec = [['nems.xforms.reverse_correlation', {}]]

    return xfspec


def tf(fitkey):
    '''
    Perform a Tensorflow fit, using Sam Norman-Haignere's CNN library

    Parameters
    ----------
    fitkey : str
        Expected format: tf.f<fitter>.i<max_iter>.s<start_conditions>.rb<count>.l<loss_function>
        Example: tf.fAdam.i1000.s20.rb10.lse
        Example translation:
            Use Adam fitter, max 1000 iterations, starting from 20 random
            initial condition, picking the best of 10 random fits, with squared error loss function

    Options
    -------
    f<fitter> : string specifying fitter, passed through to TF {'adam': 'Adam', 'gd': 'GradientDescent'}
    i<N> : Set maximum iterations to N, any positive integer.
    s<S> : Initialize with S random seeds, pick the best performer across
           the entire fit set.
    rb<N>: Picks the best of multiple random fits
    n    : Use modelspec initialized by NEMS
    l<S> : Specify the loss function {'se': 'squared_error', 'p': 'poisson', 'nmse': 'nmse', 'nmses': 'nmse_shrinkage'}
    e<N> : Specify the number of early stopping steps, i.e. the consecutive number of failed conditions before
           early stopping.
    et<N>: Specify the tolerance for early stopping. The value should be an integer, and the tolerance will be
           10 to the power of said negative integer.
    lr<N>e<N>: Specify the learning rate. The value should be two integers separated by the letter "e". The first
               integer will be multiplied by 10 raised to the negative second integer. Ex: lr5e2 = 0.05
    d<S> : String specifying the distribution with which to initialize the layers. Only used if .n not passed
    '''

    options = _extract_options(fitkey)

    max_iter = 2000
    fitter = 'Adam'
    init_count = 1
    use_modelspec_init = False
    pick_best = False
    rand_count = 0
    loss_type = 'squared_error'
    early_stopping_steps = 5
    early_stopping_tolerance = 1e-5
    learning_rate = 0.01
    distr = 'norm'

    for op in options:
        if op[:1] == 'i':
            if len(op[1:]) == 0:
                max_iter = None
            else:
                max_iter = int(op[1:])
        elif op[:1] == 'f':
            fitter = op[1:]
            if fitter == 'adam':
                fitter = 'Adam'
            elif fitter == 'gd':
                fitter = 'GradientDescent'
        elif op[:1] == 's':
            init_count = int(op[1:])
        elif op[:1] == 'n':
            use_modelspec_init = True
        elif op == 'b':
            pick_best = True
        elif op.startswith('rb'):
            if len(op) == 2:
                rand_count = 10
            else:
                rand_count = int(op[2:])
            pick_best = True
        elif op.startswith('lr'):
            learning_rate = op[2:]
            if 'e' in learning_rate:
                base, exponent = learning_rate.split('e')
                learning_rate = int(base) * 10 ** -int(exponent)
            else:
                learning_rate = int(learning_rate)
        elif op.startswith('r'):
            if len(op) == 1:
                rand_count = 10
            else:
                rand_count = int(op[1:])
        elif op[:1] == 'l':
            loss_type = op[1:]
            if loss_type == 'se':
                loss_type = 'squared_error'
            if loss_type == 'p':
                loss_type = 'poisson'
            if loss_type == 'nmse':
                loss_type = 'nmse'
            if loss_type == 'nmses':
                loss_type = 'nmse_shrinkage'
        elif op.startswith('et'):
            early_stopping_tolerance = 1 * 10 ** -int(op[2:])
        elif op[:1] == 'e':
            early_stopping_steps = int(op[1:])
        elif op[:1] == 'd':
            distr = op[1:]
            if distr == 'gu':
                distr = 'glorot_uniform'
            elif distr == 'heu':
                distr = 'he_uniform'

    xfspec = []
    if rand_count > 0:
        xfspec.append(['nems.initializers.rand_phi', {'rand_count': rand_count}])

    xfspec.append(['nems.xforms.fit_wrapper',
                   {
                       'max_iter': max_iter,
                       'use_modelspec_init': use_modelspec_init,
                       'optimizer': fitter,
                       'cost_function': loss_type,
                       'init_count': init_count,
                       'fit_function': 'nems.tf.cnnlink.fit_tf',
                       'early_stopping_steps': early_stopping_steps,
                       'early_stopping_tolerance': early_stopping_tolerance,
                       'learning_rate': learning_rate,
                       'distr': distr,
                   }])
    
    if pick_best:
        xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

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
               any positive integer. Default=[10**-4]
    SN,N,... : Fit model indices N, N... for each N given,
               where N is any positive integer or zero. May be provided
               multiple times to iterate over several successive subsets.
    tiN : Perform N per-tolerance-level iterations, where N is any
          positive integer. Default=50
    fiN : Perform N per-fit iterations, where N is any positive integer.
          Default=10
    b : choose best fit_idx based on mse_fit (relevant only when multiple
        initial conditions)
    '''

    # TODO: Support nfold and state fits for fit_iteratively?
    #       And epoch to go with state.
    options = _extract_options(fitkey)
    tolerances, module_sets, fit_iter, tol_iter, fitter, choose_best, fast_eval = \
        _parse_iter(options)

    if 'pop' in options:
        xfspec = [['nems.analysis.fit_pop_model.fit_population_iteratively',
                   {'module_sets': module_sets, 'fitter': fitter,
                    'tolerances': tolerances, 'tol_iter': tol_iter,
                    'fit_iter': fit_iter}]]

    else:
        xfspec = [['nems.xforms.fit_iteratively',
                   {'module_sets': module_sets, 'fitter': fitter,
                    'tolerances': tolerances, 'tol_iter': tol_iter,
                    'fit_iter': fit_iter}]]
    if choose_best:
        xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

    return xfspec


def _extract_options(fitkey):
    if fitkey == 'basic' or fitkey == 'iter':
        # empty options (i.e. just use defaults)
        options = []
    else:
        chunks = escaped_split(fitkey, '.')
        options = chunks[1:]
    return options


def _parse_basic(options):
    '''Options specific to basic.'''
    max_iter = 1000
    tolerance = 1e-7
    fitter = 'scipy_minimize'
    choose_best = False
    fast_eval = False
    rand_count = 1
    for op in options:
        if op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            max_iter = int(re.match(pattern, op).group(1))
        elif op.startswith('t'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            tolerance = 10**tolpower
        elif op == 'cd':
            fitter = 'coordinate_descent'
        elif op == 'b':
            choose_best = True
        elif op.startswith('rb'):
            if len(op) == 2:
                rand_count = 10
            else:
                rand_count = int(op[2:])
            choose_best = True
        elif op == 'f':
            fast_eval = True

    return max_iter, tolerance, fitter, choose_best, fast_eval, rand_count


def _parse_iter(options):
    '''Options specific to iter.'''
    tolerances = []
    module_sets = []
    fit_iter = 10
    tol_iter = 50
    fitter = 'scipy_minimize'
    choose_best = False
    fast_eval = False
    for op in options:
        if op.startswith('ti'):
            tol_iter = int(op[2:])
        elif op.startswith('fi'):
            fit_iter = int(op[2:])
        elif op.startswith('T'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            nums = op.replace('d', '.').replace('\\', '')
            powers = [float(i) for i in nums[1:].split(',')]
            tolerances.extend([10**(-1*p) for p in powers])
        elif op.startswith('S'):
            indices = [int(i) for i in op[1:].split(',')]
            module_sets.append(indices)
        elif op == 'cd':
            fitter = 'coordinate_descent'
        elif op == 'b':
            choose_best = True
        elif op == 'f':
            fast_eval = True

    if not tolerances:
        tolerances = None
    if not module_sets:
        module_sets = None

    return tolerances, module_sets, fit_iter, tol_iter, fitter, choose_best, fast_eval
