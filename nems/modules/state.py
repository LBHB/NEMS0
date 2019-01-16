"""
modules/state.py

functions for applying state-related transformations
"""

import numpy as np


def state_dc_gain(rec, i='pred', o='pred', s='state', g=None, d=0):
    '''
    Linear DC/gain for each state applied to each predicted channel

    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    g - gain to scale s by
    d - dc to offset by
    '''

    fn = lambda x: np.matmul(g, rec[s]._data) * x + np.matmul(d, rec[s]._data)

    return [rec[i].transform(fn, o)]

def state_gain(rec, i='pred', o='pred', s='state', g=None):
    '''
    Linear gain for each state applied to each predicted channel

    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    g - gain to scale s by
    '''

    fn = lambda x: np.matmul(g, rec[s]._data) * x

    return [rec[i].transform(fn, o)]


def state_segmented(rec, i='pred', o='pred', s='state'):
    '''
    2-segment linear DC/gain for each state applied to each predicted channel

    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    TODO: define paramters, specify correct transformation
    '''
    raise NotImplementedError

    fn = lambda x: np.matmul(g, rec[s]._data) * x + np.matmul(d, rec[s]._data)

    return [rec[i].transform(fn, o)]


def state_order2(rec, i='pred', o='pred', s='state'):
    '''
    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    TODO: define paramters, specify correct transformation
    '''
    raise NotImplementedError

    fn = lambda x: np.matmul(g, rec[s]._data) * x + np.matmul(d, rec[s]._data)

    return [rec[i].transform(fn, o)]


def state_weight(rec, i='pred', o='pred', s='state', g=None, d=0):
    '''
    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    g - gain to weight s by before weighting input channel
    d - dc to offset by

    o = sum_i(g_ij * s_ij * i_ij) + d*sj
    '''

    fn = lambda x: np.sum(np.matmul(g, rec[s]._data) * x,
                          axis=0, keepdims=True) + np.matmul(d, rec[s]._data)
    return [rec[i].transform(fn, o)]
