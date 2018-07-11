import logging
import re

log = logging.getLogger(__name__)


def init(kw):
    ops = kw.split('.')[1:]
    st = False
    tolerance = 10**-5.5

    for op in ops:
        if op == 'st':
            st = True
        elif op.startswith('t'):
            # 'd' stands in for '.' in floating points
            num = op.replace(',', '.')
            tolpower = float(num[1:])*(-1)
            tolerance = 10**tolpower

    if st:
        return [['nems.xforms.fit_state_init', {'tolerance': tolerance}]]
    else:
        return [['nems.xforms.fit_basic_init', {'tolerance': tolerance}]]


# TOOD: Maybe these should go in fitters instead?
#       Not really initializers, but really fitters either.
def jk(kw):
    ops = kw.split('.')[1:]
    jk_kwargs = {}
    do_split = False
    log.info("setting up n-fold fitting...")

    for op in ops:
        if op.startswith('nf'):
            jk_kwargs['njacks'] = int(op[2:])
        elif op == 'm':
            do_split = True
        elif op.startswith('ep'):
            pattern = re.compile(r'^ep(\w{1,})$')
            jk_kwargs['epoch_name'] = re.match(pattern, op).group(1)

    if do_split:
        xfspec = [['nems.xforms.split_for_jackknife', jk_kwargs]]
    else:
        xfspec = [['nems.xforms.mask_for_jackknife', jk_kwargs]]
    xfspec.append(['nems.xforms.jackknifed_fit', {}])

    return xfspec


def rand(kw):
    ops = kw.split('.')[1:]
    nt_kwargs = {}

    for op in ops:
        if op.startswith('nt'):
            nt_kwargs['ntimes'] = int(op[2:])

    return [['nems.xforms.random_sample_fit', nt_kwargs]]
