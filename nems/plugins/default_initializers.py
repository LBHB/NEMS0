import logging
import re

from nems.utils import escaped_split

log = logging.getLogger(__name__)


def init(kw):
    ops = escaped_split(kw, '.')[1:]
    st = False
    tolerance = 10**-5.5
    norm_fir = False
    fit_sig = 'resp'

    for op in ops:
        if op == 'st':
            st = True
        elif op=='psth':
            fit_sig = 'psth'
        elif op.startswith('t'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            tolerance = 10**tolpower
        elif op == 'L2f':
            norm_fir = True

    if st:
        return [['nems.xforms.fit_state_init', {'tolerance': tolerance,
                                                'fit_sig': fit_sig}]]
    else:
        return [['nems.xforms.fit_basic_init', {'tolerance': tolerance,
                                                'norm_fir': norm_fir}]]


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
    log.info("setting up n-fold fitting...")

    for op in ops:
        if op.startswith('nf'):
            jk_kwargs['njacks'] = int(op[2:])
        elif op == 'm':
            do_split = True
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
    """
    ops = kw.split('.')[1:]
    norm_method = 'meanstd'
    for op in ops:
        if op == 'ms':
            norm_method = 'meanstd'
        elif op == 'mm':
            norm_method = 'minmax'

    return [['nems.xforms.normalize_stim', {'norm_method': norm_method}]]

