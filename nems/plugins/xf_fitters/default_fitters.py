import logging
log = logging.getLogger(__name__)


def basic(fitkey):
    # prefit strf
    xfspec = [['nems.xforms.fit_basic_init', {}],
              ['nems.xforms.fit_basic', {}],
              ['nems.xforms.predict',    {}]]
    return xfspec


fit01 = basic  # add fit01 as alias for basic


def iter(fitkey):
    xfspec = [['nems.xforms.fit_iter_init', {}],
              _parse_fititer(fitkey),
              ['nems.xforms.predict', {}]]
    return xfspec


fititer = iter


def _parse_fititer(fit_keyword):
    # ex: iter01-T4-T6-S0x1-S0x1x2x3-ti50-fi20
    # fitter: scipy_minimize; tolerances: [1e-4, 1e-6]; s
    # subsets: [[0,1], [0,1,2,3]]; tol_iter: 50; fit_iter: 20;
    # Note that order does not matter except for starting with
    # 'fititer<some number>' to specify the analysis and fit algorithm
    chunks = fit_keyword.split('-')

    fit = chunks[0]
    if fit.endswith('01'):
        fitter = 'scipy_minimize'
    if fit.endswith('02'):
        fitter = 'coordinate_descent'
    else:
        fitter = 'scipy_minimize'

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
        elif c == 'cd':
            fitter = 'coordinate_descent'
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
