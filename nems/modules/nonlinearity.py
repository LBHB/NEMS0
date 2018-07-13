import cProfile
import numpy as np
from numpy import exp


def _logistic_sigmoid(x, base, amplitude, shift, kappa):
    ''' This "logistic" function only has a single negative exponent '''
    return base + amplitude/(1 + exp(-(x-shift)/kappa))


def logistic_sigmoid(rec, i, o, base, amplitude, shift, kappa):

    fn = lambda x: _logistic_sigmoid(x, base, amplitude, shift, kappa)
    return [rec[i].transform(fn, o)]


def _tanh(x, base, amplitude, shift, kappa):
    return base + (0.5 * amplitude) * (1 + np.tanh(kappa * (x - shift)))


def tanh(rec, i, o, base, amplitude, shift, kappa):
    fn = lambda x : _tanh(x, base, amplitude, shift, kappa)
    return [rec[i].transform(fn, o)]


def _quick_sigmoid(x, base, amplitude, shift, kappa):
    y = kappa * (x - shift)
    return base + (0.5 * amplitude) * (1 + y / np.sqrt(1 + np.square(y)))


def quick_sigmoid(rec, i, o, base, amplitude, shift, kappa):
    fn = lambda x : _quick_sigmoid(x, base, amplitude, shift, kappa)
    return [rec[i].transform(fn, o)]


def _double_exponential(x, base, amplitude, shift, kappa):
    # Apparently, numpy is VERY slow at taking the exponent of a negative number
    # https://github.com/numpy/numpy/issues/8233
    # The correct way to avoid this problem is to install the Intel Python Packages:
    # https://software.intel.com/en-us/distribution-for-python
    return base + amplitude * exp(-exp(np.array(-exp(kappa)) * (x - shift)))


def double_exponential(rec, i, o, base, amplitude, shift, kappa):
    '''
    A double exponential applied to all channels of a single signal.
       rec        Recording object
       i          Input signal name
       o          Output signal name
       base       Y-axis height of the center of the sigmoid
       amplitude  Y-axis distance from ymax asymptote to ymin asymptote
       shift      Centerpoint of the sigmoid along x axis
       kappa      Sigmoid curvature. Larger numbers mean steeper slopes.
    We take exp(kappa) to ensure it is always positive.
    '''
    fn = lambda x : _double_exponential(x, base, amplitude, shift, kappa)
    # fn = lambda x : _quick_sigmoid(x, base, amplitude, shift, kappa)
    # fn = lambda x : _tanh(x, base, amplitude, shift, kappa)
    # fn = lambda x : _logistic_sigmoid(x, base, amplitude, shift, kappa)
    return [rec[i].transform(fn, o)]


def _dlog(x, offset):

    # soften effects of more extreme offsets
    inflect = 2
    if offset > inflect:
        adjoffset = inflect + (offset-inflect) / 50
    elif offset < -inflect:
        adjoffset = -inflect + (offset + inflect) / 50
    else:
        adjoffset = offset

    d = 10.0**adjoffset
    zeroer = 0
    zbt = 0
    y = x.copy()

    # avoid nan-related warning
    out = ~np.isnan(y)
    out[out] = y[out] < zbt

    y[out] = zbt
    y = y - zbt

    return np.log((y + d) / d) + zeroer


def dlog(rec, i, o, offset):

    fn = lambda x : _dlog(x, offset)

    return [rec[i].transform(fn, o)]

