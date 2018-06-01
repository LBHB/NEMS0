import numpy as np
import re


def _one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


def _wc_helper(prefix, kw):
    regexp = '^{prefix}(\d{{1,}})x(\d{{1,}})$'.format(prefix=prefix)
    pattern = re.compile(regexp)
    parsed = re.match(pattern, kw)
    n_inputs = int(parsed[1])
    n_outputs = int(parsed[2])

    return n_inputs, n_outputs


def wc(kw):
    '''
    Parses the default modulespec for basic channel weighting.

    Parameter
    ---------
    kw : string
        A string of the form: wc{n_inputs}x{n_outputs}
    '''
    n_inputs, n_outputs = _wc_helper('wc', kw)

    p_coefficients = {
        'mean': np.zeros((n_outputs, n_inputs))+0.01,
        'sd': np.ones((n_outputs, n_inputs)),
    }

    template = {
        'fn': 'nems.modules.weight_channels.basic',
        'fn_kwargs': {'i': 'pred', 'o': 'pred'},
        'prior': {
            'coefficients': ('Normal', p_coefficients),
        }
    }

    return template


def wcc(kw):
    '''
    Parses the default modulespec for basic channel weighting.
    Designed for n_outputs >= n_inputs

    Parameter
    ---------
    kw : string
        A string of the form: wcc{n_inputs}x{n_outputs}
    '''
    n_inputs, n_outputs = _wc_helper('wcc', kw)

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

    template = {
        'fn': 'nems.modules.weight_channels.basic',
        'fn_kwargs': {'i': 'pred', 'o': 'pred'},
        'prior': {
            'coefficients': ('Normal', p_coefficients),
        }
    }

    return template


def wccn(kw):
    '''
    Parses the default modulespec for basic channel weighting.
    Designed for n_outputs >= n_inputs

    Parameter
    ---------
    kw : string
        A string of the form: wccn{n_inputs}x{n_outputs}
    '''
    n_inputs, n_outputs = _wc_helper('wccn', kw)

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

    template = {
        'fn': 'nems.modules.weight_channels.basic',
        'fn_kwargs': {'i': 'pred', 'o': 'pred'},
        'norm': {'type': 'minmax', 'recalc': 0, 'd': np.zeros([n_outputs, 1]),
                 'g': np.ones([n_outputs, 1])},
        'prior': {
            'coefficients': ('Normal', p_coefficients),
        }
    }

    return template


def wcg(kw):
    '''
    Parses the default modulespec for gaussian channel weighting.

    Parameter
    ---------
    kw : string
        A string of the form: wcg{n_inputs}x{n_outputs}
    '''
    n_inputs, n_outputs = _wc_helper('wcg', kw)

    # Generate evenly-spaced filter centers for the starting points
    mean = np.arange(n_outputs + 1)/(n_outputs + 1)
    mean = mean[1:]
    sd = np.full_like(mean, 1/n_outputs)

    mean_prior_coefficients = {
        'mean': mean,
        'sd': np.ones_like(mean),
    }
    sd_prior_coefficients = {'sd': sd}

    template = {
        'fn': 'nems.modules.weight_channels.gaussian',
        'fn_kwargs': {'i': 'pred', 'o': 'pred', 'n_chan_in': n_inputs},
        'fn_coefficients': 'nems.modules.weight_channels.gaussian_coefficients',
        'prior': {
            'mean': ('Normal', mean_prior_coefficients),
            'sd': ('HalfNormal', sd_prior_coefficients),
        },
    }

    return template


def wcgn(kw):
    '''
    Parses the default modulespec for gaussian channel weighting.

    Parameter
    ---------
    kw : string
        A string of the form: wcgn{n_inputs}x{n_outputs}
    '''
    n_inputs, n_outputs = _wc_helper('wcgn', kw)

    # Generate evenly-spaced filter centers for the starting points
    mean = np.arange(n_outputs + 1)/(n_outputs + 1)
    mean = mean[1:]
    sd = np.full_like(mean, 1/n_outputs)

    mean_prior_coefficients = {
        'mean': mean,
        'sd': np.ones_like(mean),
    }
    sd_prior_coefficients = {'sd': sd}

    template = {
        'fn': 'nems.modules.weight_channels.gaussian',
        'fn_kwargs': {'i': 'pred', 'o': 'pred', 'n_chan_in': n_inputs},
        'fn_coefficients': 'nems.modules.weight_channels.gaussian_coefficients',
        'norm': {'type': 'minmax', 'recalc': 0, 'd': np.zeros([n_outputs, 1]),
                 'g': np.ones([n_outputs, 1])},
        'prior': {
            'mean': ('Normal', mean_prior_coefficients),
            'sd': ('HalfNormal', sd_prior_coefficients),
        },
    }

    return template


def fir(kw):
    '''
    Generate and register default modulespec for basic channel weighting

    Parameters
    ----------
    kw : str
        A string of the form: fir{n_outputs}x{n_coefs}.
    '''
    pattern = re.compile(r'^fir(\d{1,})x(\d{1,})x?(\d{1,})?$')
    parsed = re.match(pattern, kw)
    n_outputs = int(parsed[1])
    n_coefs = int(parsed[2])
    n_banks = parsed[3]  # will be None if not given in keyword string

    p_coefficients = {
        'mean': np.zeros((n_outputs, n_coefs)),
        'sd': np.ones((n_outputs, n_coefs)),
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
                          'bank_count': int(n_banks)},
            'prior': {
                'coefficients': ('Normal', p_coefficients),
            }
        }

    return template


def lvl(kw):
    ''' TODO: this doc
    format: r'^lvl(\d{1,})$'
    '''
    pattern = re.compile(r'^lvl(\d{1,})$')
    parsed = re.match(pattern, kw)
    n_shifts = int(parsed[1])

    template = {
        'fn': 'nems.modules.levelshift.levelshift',
        'fn_kwargs': {'i': 'pred', 'o': 'pred'},
        'prior': {'level': ('Normal', {'mean': np.zeros([n_shifts, 1]),
                                       'sd': np.ones([n_shifts, 1])})}
        }

    return template


def stp(kw):
    ''' TODO: this doc
    format: r'^stp([z,n]{0,})(\d{1,})$'
    '''
    pattern = re.compile(r'^stp([z,n]{0,})(\d{1,})$')
    parsed = re.match(pattern, kw)
    options = parsed[1]
    n_synapse = int(parsed[2])

    if 'z' in options:
        u_mean = [0.02]*n_synapse
        tau_mean = [0.05]*n_synapse
    else:
        u_mean = [0.01]*n_synapse
        tau_mean = [0.04]*n_synapse
    u_sd = u_mean

    if n_synapse == 1:
        # TODO:
        # @SVD: stp1 had this as 0.01, all others 0.05. intentional?
        #       if not can just simplify this to the part within the else:
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

    if 'n' in options:
        d = np.array([0]*n_synapse)
        g = np.array([1]*n_synapse)
        template['norm'] = {'type': 'minmax', 'recalc': 0, 'd': d, 'g': g}

    return template


def dexp(kw):
    ''' TODO: this doc
    format: r'^dexp(\d{1,})$'
    '''
    pattern = re.compile(r'^dexp(\d{1,})$')
    parsed = re.match(pattern, kw)
    n_dims = int(parsed[1])

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
    ''' TODO: this doc
    format: r'^qsig(\d{1,})$'
    '''
    pattern = re.compile(r'^qsig(\d{1,})$')
    parsed = re.match(pattern, kw)
    n_dims = int(parsed[1])

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
    ''' TODO: this doc
        NOTE: these priors are typically overwritten by
              nems.initializers.init_logsig
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
    ''' TODO: this doc
    format: r'^tanh(\d{1,})$'
    '''
    pattern = re.compile(r'^tanh(\d{1,})$')
    parsed = re.match(pattern, kw)
    n_dims = int(parsed[1])

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
    ''' TODO this doc
    format: r'^dlog(n?)(\d{0,})$'
    '''
    pattern = re.compile(r'^dlog(n?)(\d{0,})$')
    parsed = re.match(pattern, kw)
    norm = parsed[1]
    chans = parsed[2]

    template = {
        'fn': 'nems.modules.nonlinearity.dlog',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'offset': ('Normal', {'mean': [0], 'sd': [2]})}
    }

    if norm:
        if not chans:
            raise ValueError('Must provide number of channels in order '
                             'to use dlog normalization: "^dlog(n?)(\d{0,})$"')
        n_chans = int(chans)
        d = np.zeros([n_chans, 1])
        g = np.ones([n_chans, 1])
        template['norm'] = {'type': 'minmax', 'recalc': 0, 'd': d, 'g': g}

    return template


def stategain(kw):
    ''' TODO: this doc
    format: r'^stategain(\d{1,})$'
    '''
    pattern = re.compile(r'^stategain(\d{1,})$')
    parsed = re.match(pattern, kw)
    n_vars = int(parsed[1])

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
    ''' TODO: this doc
    format: r'^rep(\d{1,})$'
    '''
    pattern = re.compile(r'^rep(\d{1,})$')
    parsed = re.match(pattern, kw)
    n_reps = int(parsed[1])

    template = {
        'fn': 'nems.modules.signal_mod.replicate_channels',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'repcount': n_reps},
        'phi': {}
        }

    return template


def mrg(kw):
    ''' TODO: this doc '''
    template = {
        'fn': 'nems.modules.signal_mod.merge_channels',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'phi': {}
        }
    return template
