"""
modules/state.py

functions for applying state-related transformations
"""

import numpy as np


def state_dc_gain(rec, i='pred', o='pred', s='state', c=None, g=None, d=0):
    '''
    Linear DC/gain for each state applied to each predicted channel

    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    c channels to ignore
    g - gain to scale s by
    d - dc to offset by
    '''
    fn = lambda x: np.matmul(g, rec[s]._data) * x + np.matmul(d, rec[s]._data)

    if c is None:
        return [rec[i].transform(fn, o)]
    else:
        mod_chans = np.setdiff1d(range(0, rec[i].shape[0]), c)
        mod_sigs = rec[i]._modified_copy(rec[i]._data[mod_chans, :])
        non_mod_data = rec[i]._data[c, :]
        mod_data = mod_sigs.transform(fn, o)._data

        if len(mod_data.shape) == 1:
            mod_data = mod_data[np.newaxis, :]
        if len(non_mod_data.shape) == 1:
            non_mod_data = non_mod_data[np.newaxis, :]

        newdata = np.zeros(rec[i].shape)
        newdata[mod_chans, :] = mod_data
        newdata[c, :] = non_mod_data
        new_signal = rec[i]._modified_copy(newdata)
        new_signal.name = o
        return [new_signal]

def state_gain(rec, i='pred', o='pred', s='state', c=None, g=None):
    '''
    Linear gain for each state applied to each predicted channel

    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    c channels to ignore
    g - gain to scale s by
    '''

    fn = lambda x: np.matmul(g, rec[s]._data) * x

    if c is None:
        return [rec[i].transform(fn, o)]
    else:
        mod_chans = np.setdiff1d(range(0, rec[i].shape[0]), c)
        mod_sigs = rec[i]._modified_copy(rec[i]._data[mod_chans, :])
        non_mod_data = rec[i]._data[c, :]
        mod_data = mod_sigs.transform(fn, o)._data

        if len(mod_data.shape) == 1:
            mod_data = mod_data[np.newaxis, :]
        if len(non_mod_data.shape) == 1:
            non_mod_data = non_mod_data[np.newaxis, :]

        newdata = np.zeros(rec[i].shape)
        newdata[mod_chans, :] = mod_data
        newdata[c, :] = non_mod_data
        new_signal = rec[i]._modified_copy(newdata)
        new_signal.name = o
        return [new_signal]


def state_sp_dc_gain(rec, i='pred', o='pred', s='state', g=None, d=0, sp=0):
    '''
    Linear spont/DC/gain for each state applied to each predicted channel

    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    g - gain to scale s by
    d - dc to offset by
    sp - spont to offset by
    '''

    fn = lambda x: np.matmul(g, rec[s]._data * rec['sp_mask']._data) * x + \
                   np.matmul(d, rec[s]._data* rec['ev_mask']._data) + \
                   np.matmul(sp, rec[s]._data* rec['sp_mask']._data)

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
