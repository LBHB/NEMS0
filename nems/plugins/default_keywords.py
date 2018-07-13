
'''
Default shorthands, or 'keywords,' for generating NEMS modelspecs on a
per-module basis.

Each keyword function is indexed by an instance of the KeywordRegistry class
(see nems.registry) by the name of the function. At runtime, when a full
keyword is used to index into the registry, the leading portion of the keyword
('kw_head' in the registry class) before the first '.' will be used to find
the appropriate function and the full keyword will be given as the first
argument to that function.

For example, the function `wc(kw)` defined in this module would be index as
`'wc'` in `example_registry`.
At runtime, `example_registry['wc.2x15.g']` would return the same result as
calling the function directly: `wc('wc.2x15.g')`.

Most (if not all) keyword functions expect at least one option to be present
after the initial '.'; some also accept additional options that are selected
as a part of the keyword string. For instance, in the previous example,
`2x15` indicates 2 inputs and 15 outputs, while `g` indicates gaussian
coefficients. Each additional option should be separated by a preceeding '.'

See individual keyword functions for a full description of their
parsing options.

'''

import re
import logging

import numpy as np

log = logging.getLogger(__name__)


def _one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


def wc(kw):
    '''
    Parses the default modulespec for basic and gaussian channel weighting.

    Parameter
    ---------
    kw : string
        A string of the form: wc.{n_inputs}x{n_outputs}.option1.option2...

    Options
    -------
    c : Used when n_outputs is greater than n_inputs (overwrites g)
    g : For gaussian coefficients (overwrites c)
    n : To apply normalization

    Note that the options are parsed in the order that they are passed
    and some overwrite each other, which means the last option takes
    precedence. For example,

    `wc.15x2.c.g` would be equivalent to `wc.15x2.g`,
    whereas `wc.15x2.g.c` would be equivalent to `wc.15x2.c`.
    '''
    options = kw.split('.')
    in_out_pattern = re.compile(r'^(\d{1,})x(\d{1,})$')
    try:
        parsed = re.match(in_out_pattern, options[1])
        n_inputs = int(parsed.group(1))
        n_outputs = int(parsed.group(2))
    except (TypeError, IndexError):
        # n_inputs x n_outputs should always follow wc.
        # TODO: Ideally would like the order to not matter like with other
        #       options but this seemed like a sensible solution for now
        #       since the information is mandatory.
        raise ValueError("Got TypeError or IndexError when attempting to parse "
                         "wc keyword.\nMake sure <in>x<out> is provided "
                         "as the first option after 'wc', e.g.: 'wc.2x15'"
                         "\nkeyword given: %s" % kw)

    if 'c' in options and 'g' in options:
        log.warning("Options 'c' and 'g' both given for weight_channels, but"
                    " are mutually exclusive. Whichever comes last will "
                    "overwrite the previous option. kw given: {}".format(kw))

    # This is the default for wc, but options might overwrite it.
    fn = 'nems.modules.weight_channels.basic'
    fn_kwargs = {'i': 'pred', 'o': 'pred'}
    p_coefficients = {'mean': np.zeros((n_outputs, n_inputs))+0.01,
                      'sd': np.ones((n_outputs, n_inputs))}
    prior = {'coefficients': ('Normal', p_coefficients)}
    normalize = False
    coefs = None

    for op in options[2:]:  # will be empty if only wc and {in}x{out}
        if op == 'c':

            if n_outputs == 1:
                p_coefficients = {
                    'mean': np.ones((n_outputs, n_inputs))/n_outputs,
                    'sd': np.ones((n_outputs, n_inputs)),
                }
            else:
                p_coefficients = {
                    'mean': np.eye(n_outputs, n_inputs),
                    'sd': np.ones((n_outputs, n_inputs)),
                }
                p_coefficients['mean'][(n_outputs-1):, :] = 1 / n_inputs

            prior = {'coefficients': ('Normal', p_coefficients)}

        elif op == 'g':

            # Generate evenly-spaced filter centers for the starting points
            fn = 'nems.modules.weight_channels.gaussian'
            fn_kwargs = {'i': 'pred', 'o': 'pred', 'n_chan_in': n_inputs}
            coefs = 'nems.modules.weight_channels.gaussian_coefficients'
            mean = np.arange(n_outputs + 1)/(n_outputs + 1)
            mean = mean[1:]
            sd = np.full_like(mean, 1/n_outputs)

            mean_prior_coefficients = {
                'mean': mean,
                'sd': np.ones_like(mean),
            }
            sd_prior_coefficients = {'sd': sd}
            prior = {'mean': ('Normal', mean_prior_coefficients),
                     'sd': ('HalfNormal', sd_prior_coefficients)}

        elif op == 'n':
            normalize = True

    if 'o' in options:
        fn = 'nems.modules.weight_channels.basic_with_offset'
        o_coefficients = {
            'mean': np.zeros((n_outputs, 1)),
            'sd': np.ones((n_outputs, 1))
        }
        prior['offset'] = ('Normal', o_coefficients)

    template = {
        'fn': fn,
        'fn_kwargs': fn_kwargs,
        'prior': prior
    }

    if normalize:
        template['norm'] = {'type': 'minmax', 'recalc': 0,
                            'd': np.zeros([n_outputs, 1]),
                            'g': np.ones([n_outputs, 1])}

    if coefs is not None:
        template['fn_coefficients'] = coefs

    return template


def fir(kw):
    '''
    Generate and register default modulespec for basic channel weighting

    Parameters
    ----------
    kw : str
        A string of the form: fir.{n_outputs}x{n_coefs}x{n_banks}

    Options
    -------
    None, but x{n_banks} is optional.
    '''
    pattern = re.compile(r'^fir\.?(\d{1,})x(\d{1,})x?(\d{1,})?$')
    parsed = re.match(pattern, kw)
    try:
        n_outputs = int(parsed.group(1))
        n_coefs = int(parsed.group(2))
        n_banks = parsed.group(3)  # None if not given in keyword string
    except TypeError:
        raise ValueError("Got a TypeError when parsing fir keyword. Make sure "
                         "keyword has the form: \n"
                         "fir.{n_outputs}x{n_coefs}x{n_banks} (banks optional)"
                         "\nkeyword given: %s" % kw)
    if n_banks is None:
        p_coefficients = {
            'mean': np.zeros((n_outputs, n_coefs)),
            'sd': np.ones((n_outputs, n_coefs)),
        }
    else:
        n_banks = int(n_banks)
        p_coefficients = {
            'mean': np.zeros((n_outputs * n_banks, n_coefs)),
            'sd': np.ones((n_outputs * n_banks, n_coefs)),
        }

    if n_coefs > 2:
        # p_coefficients['mean'][:, 1] = 1
        # p_coefficients['mean'][:, 2] = -0.5
        pass
    else:
        p_coefficients['mean'][:, 0] = 1

    if n_banks is None:
        template = {
            'fn': 'nems.modules.fir.basic',
            'fn_kwargs': {'i': 'pred', 'o': 'pred'},
            'prior': {
                'coefficients': ('Normal', p_coefficients),
            }
        }
    else:
        template = {
            'fn': 'nems.modules.fir.filter_bank',
            'fn_kwargs': {'i': 'pred', 'o': 'pred',
                          'bank_count': n_banks},
            'prior': {
                'coefficients': ('Normal', p_coefficients),
            }
        }

    return template


def lvl(kw):
    '''
    Generate and register default modulespec for the levelshift module.

    Parameters
    ----------
    kw : str
        Expected format: r'^lvl\.(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^lvl\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_shifts = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got a TypeError when parsing lvl keyword, "
                         "make sure keyword has the form: \n"
                         "lvl.{n_shifts}.\n"
                         "keyword given: %s" % kw)

    template = {
        'fn': 'nems.modules.levelshift.levelshift',
        'fn_kwargs': {'i': 'pred', 'o': 'pred'},
        'prior': {'level': ('Normal', {'mean': np.zeros([n_shifts, 1]),
                                       'sd': np.ones([n_shifts, 1])})}
        }

    return template


def stp(kw):
    '''
    Generate and register modulespec for short_term_plasticity module.

    Parameters
    ----------
    kw : str
        Expected format: r'^stp\.?(\d{1,})\.?([z,b,n]*)$'

    Options
    -------
    z : Change prior conditions (see function body)
    b : Set bounds on 'tau' to be greater than or equal to 0
    n : Apply normalization
    '''
    pattern = re.compile(r'^stp\.?(\d{1,})\.?([z,b,n,\.]*)$')
    parsed = re.match(pattern, kw)
    try:
        n_synapse = int(parsed.group(1))
    except (TypeError, IndexError):
        raise ValueError("Got TypeError or IndexError while parsing stp "
                         "keyword,\nmake sure keyword is of the form: \n"
                         "stp.{n_synapse}.option1.option2...\n"
                         "keyword given: %s" % kw)
    options = parsed.group(2).split('.')

    # Default values, may be overwritten by options
    u_mean = [0.01]*n_synapse
    tau_mean = [0.04]*n_synapse
    normalize = False
    bounds = False

    for op in options:
        if op == 'z':
            u_mean = [0.02]*n_synapse
            tau_mean = [0.05]*n_synapse
        elif op == 'n':
            normalize = True
        elif op == 'b':
            bounds = True

    u_sd = u_mean
    if n_synapse == 1:
        # TODO:
        # @SVD: stp1 had this as 0.01, all others 0.05. intentional?
        #       if not can just set tau_sd = [0.05]*n_synapse
        tau_sd = u_sd
    else:
        tau_sd = [0.05]*n_synapse

    template = {
        'fn': 'nems.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'crosstalk': 0},
        'prior': {'u': ('Normal', {'mean': u_mean, 'sd': u_sd}),
                  'tau': ('Normal', {'mean': tau_mean, 'sd': tau_sd})}
        }

    if normalize:
        d = np.array([0]*n_synapse)
        g = np.array([1]*n_synapse)
        template['norm'] = {'type': 'minmax', 'recalc': 0, 'd': d, 'g': g}

    if bounds:
        template['bounds'] = {'tau': (0, None)}

    return template


def dexp(kw):
    '''
    Generate and register modulespec for double_exponential module.

    Parameters
    ----------
    kw : str
        Expected format: r'^dexp\.?(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^dexp\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_dims = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got TypeError while parsing dexp keyword,\n"
                         "make sure keyword is of the form: \n"
                         "dexp.{n_dims}\nkeyword given: %s" % kw)

    base_mean = np.zeros([n_dims, 1]) if n_dims > 1 else np.array([0])
    base_sd = np.ones([n_dims, 1]) if n_dims > 1 else np.array([1])
    amp_mean = base_mean + 0.2
    amp_sd = base_mean + 0.1
    shift_mean = base_mean
    shift_sd = base_sd
    kappa_mean = base_mean
    kappa_sd = amp_sd

    template = {
        'fn': 'nems.modules.nonlinearity.double_exponential',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': base_mean, 'sd': base_sd}),
                  'amplitude': ('Normal', {'mean': amp_mean, 'sd': amp_sd}),
                  'shift': ('Normal', {'mean': shift_mean, 'sd': shift_sd}),
                  'kappa': ('Normal', {'mean': kappa_mean, 'sd': kappa_sd})}
        }

    return template


def qsig(kw):
    '''
    Generate and register modulespec for quick_sigmoid module.

    Parameters
    ----------
    kw : str
        Expected format: r'^qsig\.?(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^qsig\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    n_dims = int(parsed.group(1))

    zeros = np.zeros([n_dims, 1]) if n_dims > 1 else np.array([0])
    base_mean = zeros + 0.1
    base_sd = base_mean
    amp_mean = zeros + 0.7
    amp_sd = zeros + 0.5
    shift_mean = zeros + 1.5
    shift_sd = zeros + 1.0
    kappa_mean = base_mean
    kappa_sd = base_mean

    template = {
        'fn': 'nems.modules.nonlinearity.quick_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': base_mean, 'sd': base_sd}),
                  'amplitude': ('Normal', {'mean': amp_mean, 'sd': amp_sd}),
                  'shift': ('Normal', {'mean': shift_mean, 'sd': shift_sd}),
                  'kappa': ('Normal', {'mean': kappa_mean, 'sd': kappa_sd})}
        }

    return template


def logsig(kw):
    '''
    Generate and registry modulespec for the logistic_sigmoid module.

    Parameters
    ----------
    kw : str
        Expected format: logsig

    Options
    -------
    None

    Note
    ----
    The priors set by this keyword are typically overwritten by
    `nems.initializers.init_logsig`.
    Additionally, this function performs no parsing at this time,
    so any keyword beginning with 'logsig.' will be equivalent.
    '''
    template = {
        'fn': 'nems.modules.nonlinearity.logistic_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Exponential', {'beta': [0.1]}),
                  'amplitude': ('Exponential', {'beta': [2.0]}),
                  'shift': ('Normal', {'mean': [1.0], 'sd': [1.0]}),
                  'kappa': ('Exponential', {'beta': [0.1]})}
        }

    return template


def tanh(kw):
    '''
    Generate and register modulespec for tanh module.

    Parameters
    ----------
    kw : str
        Expected format: r'^tanh\.?(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^tanh\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_dims = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got TypeError while parsing tanh keyword,\n"
                         "make sure keyword is of the form: \n"
                         "tanh.{n_dims} \n"
                         "keyword given: %s" % kw)

    zeros = np.zeros([n_dims, 1]) if n_dims > 1 else np.array([0])
    ones = np.ones([n_dims, 1]) if n_dims > 1 else np.array([1])
    base_mean = zeros
    base_sd = ones
    amp_mean = zeros + 0.2
    amp_sd = ones
    shift_mean = zeros
    shift_sd = ones
    kappa_mean = zeros
    kappa_sd = zeros + 0.1

    template = {
        'fn': 'nems.modules.nonlinearity.tanh',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': base_mean, 'sd': base_sd}),
                  'amplitude': ('Normal', {'mean': amp_mean, 'sd': amp_sd}),
                  'shift': ('Normal', {'mean': shift_mean, 'sd': shift_sd}),
                  'kappa': ('Normal', {'mean': kappa_mean, 'sd': kappa_sd})}
    }

    return template


def dlog(kw):
    '''
    Generate and register modulespec for dlog module.

    Parameters
    ----------
    kw : str
        Expected format: r'^dlog(\.n\d{1,})?$'

    Options
    -------
    nN : Apply normalization for the given number of channels, N.
         E.g. `n18` or `n2`

    Note
    ----
    The normalization option for this function differs from the typical
    standalone 'n' because the number of channels is only needed if
    normalization is used - otherwise, only 'dlog' is required since the
    number of channels would be redundant information.
    '''
    pattern = re.compile(r'^dlog(\.?n\d{1,})?\.?([f, \.]*)$')
    parsed = re.match(pattern, kw)
    norm = parsed.group(1)
    options = parsed.group(2).split('.')
    if norm is not None:
        chans = int(norm.strip('.')[1:])  # skip leading .n
    else:
        chans = 0

    offset = ('f' in options)

    template = {
        'fn': 'nems.modules.nonlinearity.dlog',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'}
    }

    if chans:
        d = np.zeros([chans, 1])
        g = np.ones([chans, 1])
        template['norm'] = {'type': 'minmax', 'recalc': 0, 'd': d, 'g': g}

    if offset:
        template['fn_kwargs']['offset'] = -1
    else:
        template['prior'] = {'offset': ('Normal', {'mean': [0], 'sd': [2]})}

    return template


def stategain(kw):
    '''
    Generate and register modulespec for the state_dc_gain module.

    Parameters
    ----------
    kw : str
        Expected format: r'^stategain\.?(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^stategain\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_vars = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got TypeError when parsing stategain keyword.\n"
                         "Make sure keyword is of the form: \n"
                         "stategain.{n_variables} \n"
                         "keyword given: %s" % kw)

    zeros = np.zeros(n_vars)
    ones = np.ones(n_vars)
    g_mean = _one_zz(n_vars-1)
    g_sd = ones
    d_mean = zeros
    d_sd = ones

    template = {
        'fn': 'nems.modules.state.state_dc_gain',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd}),
                  'd': ('Normal', {'mean': d_mean, 'sd': d_sd})}
        }

    return template


def rep(kw):
    '''
    Generate and register modulespec for replicate_channels module.

    Parameters
    ----------
    kw : str
        Expected format: r'^rep\.?(\d{1,})$'

    Options
    -------
    None
    '''
    pattern = re.compile(r'^rep\.?(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_reps = int(parsed.group(1))
    except TypeError:
        raise ValueError("Got TypeError while parsing rep keyword. \n"
                         "Make sure keyword is of the form: \n"
                         "rep.{n_reps} \n"
                         "keyword given: %s" % kw)

    template = {
        'fn': 'nems.modules.signal_mod.replicate_channels',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'repcount': n_reps},
        'phi': {}
        }

    return template


def mrg(kw):
    '''
    Generate and register modulespec for merge_channels module.

    Parameters
    ----------
    kw : str
        Expected format: mrg

    Options
    -------
    None

    Note
    ----
    This keyword function performs no parsing. It always returns
    the same modulespec, so any keyword beginning with 'mrg.' is equivalent.
    '''
    template = {
        'fn': 'nems.modules.signal_mod.merge_channels',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'phi': {}
        }
    return template
