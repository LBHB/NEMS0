# -*- coding: utf-8 -*-
import numpy as np

def corrcoef(result, pred_name='pred', resp_name='resp'):
    '''
    Given the evaluated data, return the mean squared error

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    pred_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording

    Returns
    -------
    cc : float
        Correlation coefficient between the prediction and response.

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> cc = corrcoef(result, 'pred', 'resp')

    Note
    ----
    This function is written to be compatible with both numeric (i.e., Numpy)
    and symbolic (i.e., Theano, TensorFlow) computation. Please do not edit
    unless you know what you're doing. (@bburan TODO: Is this still true?)
    '''
    pred = result[pred_name]._data
    resp = result[resp_name]._data
    ff = np.isfinite(pred) & np.isfinite(resp)
    if np.sum(ff) == 0:
        return 0
    else:
        cc = np.corrcoef(pred[ff], resp[ff])
        return cc[0, 1]
