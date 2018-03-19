# This dict maps keywords to fragments of a modelspec
import numpy as np

defaults = {}


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
        'mean': np.zeros((n_outputs, n_inputs)),
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


def defkey_wcg(n_inputs, n_outputs):
    '''
    Generate and register default modulespec for gaussian channel weighting

    Parameters
    ----------
    n_outputs : int
        Number of output channels.

    Note
    ----
    Gaussian channel weighting does not need to know the number of input
    channels to work properly.
    '''
    name = 'wcg{}x{}'.format(n_inputs, n_outputs)

    # Generate evenly-spaced filter centers for the starting pints
    mean = np.arange(n_outputs + 2)/n_outputs
    mean = mean[1:-1]
    sd = 1/n_outputs

    mean_prior_coefficients = {
        'mean': mean,
        'sd': np.ones_like(mean),
    }
    sd_prior_coefficients = {'sd': sd}

    template = {
        'fn': 'nems.modules.weight_channels.gaussian',
        'fn_kwargs': {'i': 'pred', 'o': 'pred'},
        'fn_coefficients': 'nems.modules.weight_channels.gaussian_coefficients',
        'prior': {
            'mean': ('Normal', mean_prior_coefficients),
            'sd': ('HalfNormal', sd_prior_coefficients),
        }
    }
    return defkey(name, template)


def defkey_fir(n_coefs, n_outputs):
    '''
    Generate and register default modulespec for basic channel weighting

    Parameters
    ----------
    n_inputs : int
        Number of input channels.
    n_outputs : int
        Number of output channels.
    '''
    name = 'fir{}x{}'.format(n_coefs, n_outputs)
    p_coefficients = {
        'mean': np.zeros((n_outputs, n_coefs)),
        'sd': np.ones((n_outputs, n_coefs)),
    }

    if n_coefs > 1:
        p_coefficients['mean'][:, 1] = 1
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


# Autogenerate some standard keywords. TODO: this should be parseable from the
# keyword name rather than requring an explicit definition for each. Port over
# the parsing code from old NEMS?
for n_inputs in (15, 18, 40):
    for n_outputs in (1, 2, 3, 4):
        defkey_wc(n_inputs, n_outputs)
        defkey_wcg(n_inputs, n_outputs)


for n_coefs in (10, 15, 18):
    defkey_fir(n_coefs, 1)

defkey_fir(10, 2)
defkey_fir(15, 2)
defkey_fir(18, 2)

defkey('lvl1',
       {'fn': 'nems.modules.levelshift.levelshift',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'level': ('Normal', {'mean': [0], 'sd': [1]})}})

defkey('dexp1',
       {'fn': 'nems.modules.nonlinearity.double_exponential',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': [0], 'sd': [1]}),
                  'amplitude': ('Normal', {'mean': [0.2], 'sd': [0.1]}),
                  'shift': ('Normal', {'mean': [0], 'sd': [1]}),
                  'kappa': ('Normal', {'mean': [0], 'sd': [0.1]})}})

defkey('qsig1',
       {'fn': 'nems.modules.nonlinearity.quick_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': [0.1], 'sd': [0.1]}),
                  'amplitude': ('Normal', {'mean': [0.7], 'sd': [0.5]}),
                  'shift': ('Normal', {'mean': [1.5], 'sd': [1.0]}),
                  'kappa': ('Normal', {'mean': [0.1], 'sd': [0.1]})}})

defkey('logsig1',
       {'fn': 'nems.modules.nonlinearity.logistic_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mean': [0], 'sd': [1]}),
                  'amplitude': ('Normal', {'mean': [0.2], 'sd': [1]}),
                  'shift': ('Normal', {'mean': [0], 'sd': [1]}),
                  'kappa': ('Normal', {'mean': [0], 'sd': [0.1]})}})

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
        'prior': {'offset': ('Normal', {'mean': [-2], 'sd': [2]})}})


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
                  'd': ('Normal', {'mean': [1,0], 'sd': [1,1]})}
        })

defkey('psth',
       {'fn': 'nems.modules.signal_mod.average_sig',
        'fn_kwargs': {'i': 'resp',
                      'o': 'pred'}
        })

