# This dict maps keywords to fragments of a modelspec
import numpy as np

defaults = {}


def _one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


def defkey(keyword, modulespec):
    '''
    Adds modulespec to the defaults keyword dictionary.
    A helper function so not every keyword mapping has to be in a single
    file and part of a very large single multiline dict.
    '''
    if keyword in defaults:
        m = "Keyword {} already defined! Choose another name."
        raise ValueError(m.format(keyword))
    defaults[keyword] = modulespec


def defkey_wc(n_inputs, n_outputs):
    '''
    Generate and register default modulespec for basic channel weighting

    Parameters
    ----------
    n_inputs : int
        Number of input channels.
    n_outputs : int
        Number of output channels.
    '''
    name = 'wc{}x{}'.format(n_inputs, n_outputs)
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
    return defkey(name, template)


def defkey_wcc(n_inputs, n_outputs):
    '''
    Generate and register default modulespec for basic channel weighting.
    wcc is designed for n_outputs>=n_inputs

    Parameters
    ----------
    n_inputs : int
        Number of input channels.
    n_outputs : int
        Number of output channels.
    '''
    name = 'wcc{}x{}'.format(n_inputs, n_outputs)
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

    template = {
        'fn': 'nems.modules.weight_channels.basic',
        'fn_kwargs': {'i': 'pred', 'o': 'pred'},
        'prior': {
            'coefficients': ('Normal', p_coefficients),
        }
    }
    return defkey(name, template)


def defkey_wccn(n_inputs, n_outputs):
    '''
    Generate and register default modulespec for basic channel weighting.
    wcc is designed for n_outputs>=n_inputs

    Parameters
    ----------
    n_inputs : int
        Number of input channels.
    n_outputs : int
        Number of output channels.
    '''
    name = 'wccn{}x{}'.format(n_inputs, n_outputs)
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
    return defkey(name, template)


def defkey_wcg(n_inputs, n_outputs):
    '''
    Generate and register default modulespec for gaussian channel weighting

    Parameters
    ----------
    n_inputs : int
        Number of input channels.
    n_outputs : int
        Number of output channels.

    Note
    ----
    Gaussian channel weighting does not need to know the number of input
    channels to work properly.
    '''
    name = 'wcg{}x{}'.format(n_inputs, n_outputs)

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
    return defkey(name, template)


def defkey_wcgn(n_inputs, n_outputs):
    '''
    Generate and register default modulespec for gaussian channel weighting

    Parameters
    ----------
    n_inputs : int
        Number of input channels.
    n_outputs : int
        Number of output channels.

    Note
    ----
    Gaussian channel weighting does not need to know the number of input
    channels to work properly.
    '''
    name = 'wcgn{}x{}'.format(n_inputs, n_outputs)

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
        'norm': {'type': 'minmax', 'recalc': 0, 'd': np.zeros([n_outputs,1]),
                 'g': np.ones([n_outputs,1])},
        'prior': {
            'mean': ('Normal', mean_prior_coefficients),
            'sd': ('HalfNormal', sd_prior_coefficients),
        },
    }
    return defkey(name, template)


def defkey_fir(n_coefs, n_outputs):
    '''
    Generate and register default modulespec for basic channel weighting

    Parameters
    ----------
    n_coefs : int
        Number of filter coefficients/taps.
    n_outputs : int
        Number of channels.
    '''
    name = 'fir{}x{}'.format(n_outputs, n_coefs)
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

    template = {
        'fn': 'nems.modules.fir.basic',
        'fn_kwargs': {'i': 'pred', 'o': 'pred'},
        'prior': {
            'coefficients': ('Normal', p_coefficients),
        }
    }
    return defkey(name, template)


def defkey_firbank(n_coefs, n_inputs, n_banks):
    '''
    Generate and register default modulespec for basic channel weighting

    Parameters
    ----------
    n_coefs : int
        Number of filter coefficients/taps.
    n_inputs : int
        Number of input channels.
    n_banks : int
        Number of output channels per input channel.
    '''
    name = 'fir{}x{}x{}'.format(n_banks, n_inputs, n_coefs)
    p_coefficients = {
        'mean': np.zeros((n_banks*n_inputs, n_coefs)),
        'sd': np.ones((n_banks*n_inputs, n_coefs)),
    }

    if n_coefs > 2:
        # p_coefficients['mean'][:, 1] = 1
        # p_coefficients['mean'][:, 2] = -0.5
        pass
    else:
        p_coefficients['mean'][:, 0] = 1

    template = {
        'fn': 'nems.modules.fir.filter_bank',
        'fn_kwargs': {'i': 'pred', 'o': 'pred', 'bank_count': n_banks},
        'prior': {
            'coefficients': ('Normal', p_coefficients),
        }
    }
    return defkey(name, template)


# Autogenerate some standard keywords. TODO: this should be parseable from the
# keyword name rather than requring an explicit definition for each. Port over
# the parsing code from old NEMS?
for n_inputs in (15, 18, 40):
    for n_outputs in (1, 2, 3, 4):
        defkey_wc(n_inputs, n_outputs)
        defkey_wcg(n_inputs, n_outputs)
        defkey_wcgn(n_inputs, n_outputs)

for n_inputs in (1, 2, 3):
    for n_outputs in (1, 2, 3, 4):
        defkey_wcc(n_inputs, n_outputs)
        defkey_wccn(n_inputs, n_outputs)

for n_outputs in range(1, 20):
    for n_coefs in (10, 15, 18, 20, 25, 30, 40):
        defkey_fir(n_coefs, n_outputs)

defkey_firbank(15, 1, 2)
defkey_firbank(15, 2, 2)

# defkey_fir(10, 2)
# defkey_fir(15, 2)
# defkey_fir(18, 2)

defkey('lvl1',
       {'fn': 'nems.modules.levelshift.levelshift',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'level': ('Normal', {'mean': np.zeros([1, 1]),
                                       'sd': np.ones([1, 1])})}
        })

defkey('lvl2',
       {'fn': 'nems.modules.levelshift.levelshift',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'level': ('Normal', {'mean': np.zeros([2, 1]),
                                       'sd': np.ones([2, 1])})}
        })

defkey('stp1',
       {'fn': 'nems.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'crosstalk': 0},
        'prior': {'u': ('Normal', {'mean': [0.01], 'sd': [0.01]}),
                  'tau': ('Normal', {'mean': [0.04], 'sd': [0.01]})}
        })

defkey('stp2',
       {'fn': 'nems.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'crosstalk': 0},
        'prior': {'u': ('Normal', {'mean': [.01, .01], 'sd': [.01, .01]}),
                  'tau': ('Normal', {'mean': [.04, .04], 'sd': [.05, .05]})},
        })
#        'bounds': {'u': ([-0.5, -0.5], [0.5, 0.5]),
#                   'tau': ([.02, .02], None)}

defkey('stp3',
       {'fn': 'nems.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'crosstalk': 0},
        'prior': {'u': ('Normal', {'mean': [.01, .01, .01],
                                   'sd': [.01, .01, .01]}),
                  'tau': ('Normal', {'mean': [.04, .04, .04],
                                     'sd': [.05, .05, .05]})}
        })

defkey('stp4',
       {'fn': 'nems.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'crosstalk': 0},
        'prior': {'u': ('Normal', {'mean': [.01, .01, .01, .01],
                                   'sd': [.01, .01, .01, .01]}),
                  'tau': ('Normal', {'mean': [.04, .04, .04, .04],
                                     'sd': [.05, .05, .05, .05]})}
        })

defkey('stp2b',
       {'fn': 'nems.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'crosstalk': 0},
        'prior': {'u': ('Normal', {'mean': [.01, .01], 'sd': [.01, .01]}),
                  'tau': ('Normal', {'mean': [.04, .04], 'sd': [.05, .05]})},
        'bounds': {'tau': ([0, 0], None)}
        })

defkey('stpz2',
       {'fn': 'nems.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'crosstalk': 0},
        'prior': {'u': ('Normal', {'mean': [.02, .02], 'sd': [.02, .02]}),
                  'tau': ('Normal', {'mean': [.05, .05], 'sd': [.05, .05]})}
        })

defkey('stpn1',
       {'fn': 'nems.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'crosstalk': 0},
        'norm': {'type': 'minmax', 'recalc': 0, 'd': [0], 'g': [1]},
        'prior': {'u': ('Normal', {'mean': [0.01], 'sd': [0.01]}),
                  'tau': ('Normal', {'mean': [0.04], 'sd': [0.01]})}
        })
defkey('stpn2',
       {'fn': 'nems.modules.stp.short_term_plasticity',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'crosstalk': 0},
        'norm': {'type': 'minmax', 'recalc': 0, 'd': np.array([[0, 0]]),
                 'g': np.array([[1, 1]])},
        'prior': {'u': ('Normal', {'mean': [.01, .01], 'sd': [.01, .01]}),
                  'tau': ('Normal', {'mean': [.04, .04], 'sd': [.05, .05]})}
        })

defkey('dexp1',
       {'fn': 'nems.modules.nonlinearity.double_exponential',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': [0], 'sd': [1]}),
                  'amplitude': ('Normal', {'mean': [0.2], 'sd': [0.1]}),
                  'shift': ('Normal', {'mean': [0], 'sd': [1]}),
                  'kappa': ('Normal', {'mean': [0], 'sd': [0.1]})}})

defkey('dexp2',
       {'fn': 'nems.modules.nonlinearity.double_exponential',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': np.zeros([2, 1]), 'sd': np.ones([2, 1])}),
                  'amplitude': ('Normal', {'mean': np.zeros([2, 1])+0.2, 'sd': np.zeros([2, 1])+0.1}),
                  'shift': ('Normal', {'mean': np.zeros([2, 1]), 'sd': np.ones([2, 1])}),
                  'kappa': ('Normal', {'mean': np.zeros([2, 1]), 'sd': np.zeros([2, 1])+0.1})}})

defkey('qsig1',
       {'fn': 'nems.modules.nonlinearity.quick_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': [0.1], 'sd': [0.1]}),
                  'amplitude': ('Normal', {'mean': [0.7], 'sd': [0.5]}),
                  'shift': ('Normal', {'mean': [1.5], 'sd': [1.0]}),
                  'kappa': ('Normal', {'mean': [0.1], 'sd': [0.1]})}})

# NOTE: Typically overwritten by nems.initializers.init_logsig
defkey('logsig1',
       {'fn': 'nems.modules.nonlinearity.logistic_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Exponential', {'beta': [0.1]}),
                  'amplitude': ('Exponential', {'beta': [2.0]}),
                  'shift': ('Normal', {'mean': [1.0], 'sd': [1.0]}),
                  'kappa': ('Exponential', {'beta': [0.1]})}})

defkey('tanh1',
       {'fn': 'nems.modules.nonlinearity.tanh',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': [0], 'sd': [1]}),
                  'amplitude': ('Normal', {'mean': [0.2], 'sd': [1]}),
                  'shift': ('Normal', {'mean': [0], 'sd': [1]}),
                  'kappa': ('Normal', {'mean': [0], 'sd': [0.1]})}})

defkey('dlog',
       {'fn': 'nems.modules.nonlinearity.dlog',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'offset': ('Normal', {'mean': [0], 'sd': [2]})}})

defkey('dlogf',
       {'fn': 'nems.modules.nonlinearity.dlog',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'offset': -1}})

defkey('dlogz',
       {'fn': 'nems.modules.nonlinearity.dlog',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'offset': ('Normal', {'mean': [0], 'sd': [2]})}})

defkey('dlogn2',
       {'fn': 'nems.modules.nonlinearity.dlog',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'norm': {'type': 'minmax', 'recalc': 0, 'd': np.array([[0, 0]]),
                 'g': np.array([[1, 1]])},
        'prior': {'offset': ('Normal', {'mean': [-2], 'sd': [2]})}})

defkey('dlogn18',
       {'fn': 'nems.modules.nonlinearity.dlog',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'norm': {'type': 'minmax', 'recalc': 0, 'd': np.zeros([18, 1]),
                 'g': np.ones([18, 1])},
        'prior': {'offset': ('Normal', {'mean': [0], 'sd': [2]})}})


""" state-related and signal manipulation/generation """

defkey('pup',
       {'fn': 'nems.modules.signal_mod.make_state_signal',
        'fn_kwargs': {'signals_in': ['pupil'],
                      'signals_permute': [],
                      'o': 'state'}
        })

defkey('stategain2',
       {'fn': 'nems.modules.state.state_dc_gain',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': [1,0], 'sd': [1,1]}),
                  'd': ('Normal', {'mean': [0,0], 'sd': [1,1]})}
        })

defkey('stategain3',
       {'fn': 'nems.modules.state.state_dc_gain',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': [1,0,0], 'sd': [1,1,1]}),
                  'd': ('Normal', {'mean': [0,0,0], 'sd': [1,1,1]})}
        })

defkey('stategain4',
       {'fn': 'nems.modules.state.state_dc_gain',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': _one_zz(3),
                                   'sd': np.ones(4)}),
                  'd': ('Normal', {'mean': np.zeros(4),
                                   'sd': np.ones(4)})}
            })

defkey('stategain5',
       {'fn': 'nems.modules.state.state_dc_gain',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': _one_zz(4),
                                   'sd': np.ones(5)}),
                  'd': ('Normal', {'mean': np.zeros(5),
                                   'sd': np.ones(5)})}
            })


defkey('stategain6',
       {'fn': 'nems.modules.state.state_dc_gain',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': _one_zz(5),
                                   'sd': np.ones(6)}),
                  'd': ('Normal', {'mean': np.zeros(6),
                                   'sd': np.ones(6)})}
            })


defkey('stategain28',
       {'fn': 'nems.modules.state.state_dc_gain',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'prior': {'g': ('Normal', {'mean': _one_zz(27),
                                   'sd': np.ones(28)}),
                  'd': ('Normal', {'mean': np.zeros(28),
                                   'sd': np.ones(28)})}
        })


defkey('psth',
       {'fn': 'nems.modules.signal_mod.average_sig',
        'fn_kwargs': {'i': 'resp',
                      'o': 'pred'}
        })

defkey('rep2',
       {'fn': 'nems.modules.signal_mod.replicate_channels',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'repcount': 2},
        'phi': {}
        })

defkey('rep3',
       {'fn': 'nems.modules.signal_mod.replicate_channels',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'repcount': 3},
        'phi': {}
        })

defkey('mrg',
       {'fn': 'nems.modules.signal_mod.merge_channels',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': 'state'},
        'phi': {}
        })
