import cProfile
import copy
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


def double_exponential(rec, i, o, base, amplitude, shift, kappa, **kwargs):
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
    """
    Log compression helper function
    :param x: input, needs to be >0, works best if x values range approximately (0, 1)
    :param offset: threshold (d = 10**offset). offset compressed for |offset|>2
    :return: y = np.log((x + d) / d)
    """

    # soften effects of more extreme offsets
    inflect = 2

    if isinstance(offset, int):
        offset = np.array([[offset]])

    adjoffset = offset.copy()
    adjoffset[offset > inflect] = inflect + (offset[offset > inflect]-inflect) / 50
    adjoffset[offset < -inflect] = -inflect + (offset[offset < -inflect]+inflect) / 50

    d = 10.0**adjoffset

    return np.log((x + d) / d)


def dlog(rec, i, o, offset, **kwargs):
    """
    Log compression with variable offset
    :param rec: recording object with signals i and o.
    :param i: input signal name (x)
    :param o: output signal name (y)
    :param offset: threshold (d)
    :return: y = np.log((x + d) / d)
    """

    fn = lambda x: _dlog(x, offset)

    return [rec[i].transform(fn, o)]


def _relu(x, offset):
    """
    Linear rectifier helper function
    :param x: input
    :param offset: threshold
    :return:  y= x-offset , if x>offset
               = 0 otherwise
    """
    y = x - offset
    y[y < 0] = 0

    return y


def relu(rec, i, o, offset, var_offset=True):
    """
    Simple linear rectifier
    :param rec: recording object with signals i and o.
    :param i: input signal name (x)
    :param o: output signal name (y)
    :param offset: threshold (d)
    :return: y = x - d , if x>d
               = 0 otherwise
    """
    fn = lambda x: _relu(x, offset)

    return [rec[i].transform(fn, o)]


def _relub(x, offset, baseline):
    """
    Linear rectifier helper function
    :param x: input
    :param offset: threshold
    :param baseline: spont
    :return:  y= x-offset , if x>offset
               = 0 otherwise
    """
    y = x - offset
    y[y < 0] = 0
    y += baseline

    return y

def relub(rec, i, o, offset, baseline):
    """
    Simple linear rectifier
    :param rec: recording object with signals i and o.
    :param i: input signal name (x)
    :param o: output signal name (y)
    :param offset: threshold (d)
    :return: y = x - d , if x>d
               = 0 otherwise
    """
    fn = lambda x: _relub(x, offset, baseline)

    return [rec[i].transform(fn, o)]


def _saturated_rectifier(x, base, amplitude, shift, kappa):
    if base - amplitude > 0:
        y = base - kappa*(x-shift)
        y[y > base] = base
        y[y < amplitude] = amplitude
    else:
        y = kappa*(x-shift) + base
        y[y < base] = base
        y[y > amplitude] = amplitude

    return y


def saturated_rectifier(rec, i, o, base, amplitude, shift, kappa):
    '''
    More complicated linear rectifier that mimics sigmoidal nonlinearities.

    Parameters:
    -----------
    base : float
        Spontaneous firing rate, the value that more negative predictions
        will be rectified to.
    amplitude : float
        Saturated firing rate, the value that more positive predictions
        will be truncated to.
    kappa : float
        Roughly neural gain, slope of the output/input relationship.
    shift : float
        Firing threshold.

    '''

    fn = lambda x: _saturated_rectifier(x, base, amplitude, kappa, shift)
    return [rec[i].transform(fn, o)]
