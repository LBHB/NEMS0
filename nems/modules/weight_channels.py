import numpy as np


#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------
def gaussian_coefficients(mean, sd, n_chan_in, **kwargs):
    '''
    Generate Gaussian-constrained coefficients vector for channel weighting

    Parameters
        mean - typically 0-1 but can be outside of only edge of Gaussian
        std - must be >0. Hard bound at 0.00001
    ----------
    TODO
    '''
    x = np.arange(n_chan_in)/n_chan_in
    mean = np.asanyarray(mean)[..., np.newaxis]
    # hard bounds, not necessary or useful even?
    # mean[mean < 0] = 0
    # mean[mean > 1] = 1
    sd = np.asanyarray(sd)[..., np.newaxis]
    # sd[sd < 0.00001] = 0.00001
    #coefficients = 1/(sd*(2*np.pi)**0.5) * np.exp(-0.5*((x-mean)/sd)**2)
    coefficients = np.exp(-0.5*((x-mean)/sd)**2)
    csum = np.sum(coefficients, axis=1, keepdims=True)
    csum[csum == 0] = 1
    coefficients /= csum
    return coefficients


#-------------------------------------------------------------------------------
# Module functions
#-------------------------------------------------------------------------------
def basic(rec, i, o, coefficients, normalize_coefs=False, **kwargs):
    '''
    Parameters
    ----------
    coefficients : 2d array (output channel x input channel weights)
        Weighting of the input channels. A set of weights are provided for each
        desired output channel. Each row in the array are the weights for the
        input channels for that given output. The length of the row must be
        equal to the number of channels in the input array
        (e.g., `x.shape[-3] == coefficients.shape[-1]`).
    '''
    if normalize_coefs:
        c = coefficients.copy()
        sc = np.sum(np.abs(c), axis=1, keepdims=True)
        sc[sc == 0] = 1
        c /= sc
        fn = lambda x: c @ x
    else:
        fn = lambda x: coefficients @ x

    return [rec[i].transform(fn, o)]


def basic_with_offset(rec, i, o, coefficients, offset, normalize_coefs=False):
    '''
    Parameters
    ----------
    coefficients : 2d array (output channel x input channel weights)
        Weighting of the input channels. A set of weights are provided for each
        desired output channel. Each row in the array are the weights for the
        input channels for that given output. The length of the row must be
        equal to the number of channels in the input array
        (e.g., `x.shape[-3] == coefficients.shape[-1]`).
    '''

    if normalize_coefs:
        c = coefficients.copy()
        sc = np.sum(np.abs(c), axis=1, keepdims=True)
        sc[sc == 0] = 1
        c /= sc
    else:
        c = coefficients

    fn = lambda x: c @ x + offset
    return [rec[i].transform(fn, o)]


def basic_with_rect(rec, i, o, coefficients, offset, normalize_coefs=False):
    '''
    Parameters
    ----------
    coefficients : 2d array (output channel x input channel weights)
        Weighting of the input channels. A set of weights are provided for each
        desired output channel. Each row in the array are the weights for the
        input channels for that given output. The length of the row must be
        equal to the number of channels in the input array
        (e.g., `x.shape[-3] == coefficients.shape[-1]`).
    '''

    if normalize_coefs:
        c = coefficients.copy()
        sc = np.sum(np.abs(c), axis=1, keepdims=True)
        sc[sc == 0] = 1
        c /= sc
    else:
        c = coefficients

    def fn(x):
        y = c @ x + offset
        y[y<0] = 0
        return y

    return [rec[i].transform(fn, o)]


def gaussian(rec, i, o, n_chan_in, mean, sd, **kw_args):
    '''
    Parameters
    ----------
    rec : recording
        Recording to transform
    i : string
        Name of input signal
    o : string
        Name of output signal
    mean : array-like (between 0 and 1)
        Centers of Gaussian channel weights
    sd : array-like
        Standard deviation of Gaussian channel weights
    '''
    coefficients = gaussian_coefficients(mean, sd, n_chan_in)
    fn = lambda x: coefficients @ x
    return [rec[i].transform(fn, o)]
