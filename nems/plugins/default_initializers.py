import logging
import re

from nems.utils import escaped_split, keyword_extract_options

log = logging.getLogger(__name__)


def init(kw):
    '''
    Initialize modelspecs in an attempt to avoid getting stuck in
    local minima.
    Written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)

    Parameter
    ---------
    kw : string
        A string of the form: init.option1.option2...

    Options
    -------
    tN : Set tolerance to 10**-N, where N is any positive integer.
    st : Remove state replication/merging before initializing.
    psth : Initialize by fitting to 'psth' intead of 'resp' (default)
    nlN : Initialize nonlinearity with verion N
        For dexp, options are {1,2} (default is 2),
            pre 11/29/18 models were fit with v1
            1: amp = np.nanstd(resp) * 3
               kappa = np.log(2 / (np.max(pred) - np.min(pred) + 1))
            2:
               amp = resp[pred>np.percentile(pred,90)].mean()
               kappa = np.log(2 / (np.std(pred)*3))
        For other nonlinearities, mode is not specified yet
    L2f : normalize fir (default false)
    .rN : initialize with N random phis drawn from priors (via init.rand_phi),
          default N=10
    .rbN : initialize with N random phis drawn from priors (via init.rand_phi),
           and pick best mse_fit, default N=10

    TODO: Optimize more, make testbed to check how well future changes apply
    to disparate datasets.

    '''

    ops = escaped_split(kw, '.')[1:]
    st = False
    tolerance = 10**-5.5
    norm_fir = False
    fit_sig = 'resp'
    nl_kw = {}
    rand_count = 0
    keep_best = False
    fast_eval = ('f' in ops)
    tf = False
    for op in ops:
        if op == 'st':
            st = True
        elif op.startswith('tf'):
            tf = True
        elif op=='psth':
            fit_sig = 'psth'
        elif op.startswith('nl'):
            nl_kw = {'nl_mode': int(op[2:])}
        elif op.startswith('t'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            tolerance = 10**tolpower
        elif op == 'L2f':
            norm_fir = True
        elif op.startswith('rb'):
            if len(op) == 2:
                rand_count = 10
            else:
                rand_count = int(op[2:])
            keep_best = True
        elif op.startswith('r'):
            if len(op) == 1:
                rand_count = 10
            else:
                rand_count = int(op[1:])
        elif op.startswith('b'):
            keep_best = True

    xfspec = []
    #if fast_eval:
    #    xfspec.append(['nems.xforms.fast_eval', {}])
    if rand_count > 0:
        xfspec.append(['nems.initializers.rand_phi', {'rand_count': rand_count}])

    if tf:
        xfspec.append(['nems.xforms.fit_wrapper',
                       {'tolerance': tolerance, 'norm_fir': norm_fir, 'nl_kw': nl_kw,
                        'fit_function': 'nems.tf.cnnlink.fit_tf_init'}])
    elif st:
        xfspec.append(['nems.xforms.fit_wrapper',
                       {'tolerance': tolerance, 'norm_fir': norm_fir,
                        'nl_kw': nl_kw, 'fit_sig': fit_sig,
                        'fit_function': 'nems.xforms.fit_state_init'}])
    else:
        xfspec.append(['nems.xforms.fit_wrapper',
                       {'tolerance': tolerance, 'norm_fir': norm_fir, 'nl_kw': nl_kw,
                        'fit_function': 'nems.xforms.fit_basic_init'}])
    if keep_best:
        xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'criterion': 'mse_fit'}])

    return xfspec


def initpop(kw):
    options = keyword_extract_options(kw)

    rnd = False
    flip_pcs = True
    start_count = 1
    usepcs = True
    for op in options:
        if op.startswith("rnd"):
            rnd = True
            if len(op)>3:
                start_count=int(op[3:])
        elif op=='nflip':
            flip_pcs = False

    if rnd:
        xfspec = [['nems.xforms.fit_wrapper',
                   {'pc_signal': 'rand_resp', 'start_count': start_count,
                    'fit_function': 'nems.analysis.fit_pop_model.init_pop_rand'}]]
    elif usepcs:
        xfspec = [['nems.xforms.fit_wrapper',
                   {'flip_pcs': flip_pcs,
                    'fit_function': 'nems.analysis.fit_pop_model.init_pop_pca'}]]
    return xfspec


# TOOD: Maybe these should go in fitters instead?
#       Not really initializers, but really fitters either.
# move to same place as sev? -- SVD
# TODO: Maybe can keep splitep and avgep as one thing?
#       Would they ever be done separately?
def timesplit(kw):
    frac = int(kw.split('.')[1][1:])*0.1
    return [['nems.xforms.split_at_time', {'fraction': frac}]]


def splitep(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    xfspec = [['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': epoch_regex}]]
    return xfspec


def avgep(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    return [['nems.xforms.average_away_stim_occurrences',
             {'epoch_regex': epoch_regex}]]


def sev(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    xfspec = [['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': epoch_regex}],
        ['nems.xforms.average_away_stim_occurrences',
         {'epoch_regex': epoch_regex}]]
    return xfspec


def aev(kw):
    xfspec= [['nems.xforms.use_all_data_for_est_and_val', 
                {}]]
    return xfspec


def sevst(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM' if not ops else ops[0]
    xfspec = [['nems.xforms.split_by_occurrence_counts',
               {'epoch_regex': epoch_regex}]]
    return xfspec


def tev(kw):
    ops = kw.split('.')[1:]

    valfrac = 0.1
    for op in ops:
        if op.startswith("vv"):
            valfrac=int(op[2:]) / 1000
        elif op.startswith("v"):
            valfrac=int(op[1:]) / 100

    xfspec = [['nems.xforms.split_at_time', {'valfrac': valfrac}]]

    return xfspec


def jk(kw):
    ops = kw.split('.')[1:]
    jk_kwargs = {}
    do_split = False
    keep_only = 0
    log.info("Setting up N-fold fitting...")
    jk_kwargs['allow_partial_epochs'] = False

    for op in ops:
        if op.startswith('nf'):
            jk_kwargs['njacks'] = int(op[2:])
        elif op == 'm':
            do_split = True
        elif op == 'p':
            jk_kwargs['allow_partial_epochs'] = True
        elif op.startswith('ep'):
            pattern = re.compile(r'^ep(\w{1,})$')
            jk_kwargs['epoch_name'] = re.match(pattern, op).group(1)
        elif op.startswith('o'):
            if len(op)>1:
                keep_only = int(op[1:])
            else:
                keep_only = 1
        elif op.startswith('bt'):
            # jackknife by time
            jk_kwargs['by_time'] = True

    if do_split:
        xfspec = [['nems.xforms.split_for_jackknife', jk_kwargs]]
    else:
        xfspec = [['nems.xforms.mask_for_jackknife', jk_kwargs]]
    if keep_only == 1:
        xfspec.append(['nems.xforms.jack_subset', {'keep_only': keep_only}])
    elif keep_only > 1:
        xfspec.append(['nems.xforms.jack_subset', {'keep_only': keep_only}])
        xfspec.append(['nems.xforms.jackknifed_fit', {}])
    else:
        xfspec.append(['nems.xforms.jackknifed_fit', {}])

    return xfspec


def rand(kw):
    ops = kw.split('.')[1:]
    nt_kwargs = {}

    for op in ops:
        if op.startswith('nt'):
            nt_kwargs['ntimes'] = int(op[2:])
        elif op.startswith('S'):
            nt_kwargs['subset'] = [int(i) for i in op[1:].split(',')]

    return [['nems.xforms.random_sample_fit', nt_kwargs]]


def norm(kw):
    """
    Normalize stim and response before splitting/fitting to support
    fitters that can't deal with big variations in values

    default is to normalize minmax (norm.mm) to fall in the range 0 to 1 (keep values
    positive to support log compression)
    norm.ms will normalize to (mean=0, std=1)
    """
    ops = kw.split('.')[1:]
    norm_method = 'minmax'
    for op in ops:
        if op == 'ms':
            norm_method = 'meanstd'
        elif op == 'mm':
            norm_method = 'minmax'

    return [['nems.xforms.normalize_sig', {'sig': 'stim', 'norm_method': norm_method}],
            ['nems.xforms.normalize_sig', {'sig': 'resp', 'norm_method': norm_method}],
            ]
