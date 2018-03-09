import numpy as np

def mse(result, pred_name='pred', resp_name='resp'):
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
    mse : float
        Mean-squared difference between the prediction and response.

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> error = mse(result, 'pred', 'resp')

    Note
    ----
    This function is written to be compatible with both numeric (i.e., Numpy)
    and symbolic (i.e., Theano, TensorFlow) computation. Please do not edit
    unless you know what you're doing. (@bburan TODO: Is this still true?)
    '''
    pred = result[pred_name]
    resp = result[resp_name]
    squared_errors = (pred-resp)**2
    return np.nanmean(squared_errors)


def nmse(result, pred_name='pred', resp_name='resp'):
    '''
    Same as MSE, but normalized by the std of the resp.
    Because it is more computationally expensive than MSE but is otherwise
    equivalent, we suggest using the MSE for fitting and use this as a
    post-fit performance metric only.
    '''
    pred = result[pred_name]
    resp = result[resp_name]
    respstd = np.nanstd(resp)
    squared_errors = (pred-resp)**2
    mse = np.nanmean(squared_errors)
    return mse / respstd
