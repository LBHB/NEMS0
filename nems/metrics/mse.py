import numpy as np
import nems.utils
import logging

log = logging.getLogger(__name__)


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
    pred = result[pred_name].as_continuous()
    resp = result[resp_name].as_continuous()
    squared_errors = (pred-resp)**2
    return np.nanmean(squared_errors)


def nmse(result, pred_name='pred', resp_name='resp'):
    '''
    Same as MSE, but normalized by the std of the resp.
    Because it is more computationally expensive than MSE but is otherwise
    equivalent, we suggest using the MSE for fitting and use this as a
    post-fit performance metric only.
    '''
    pred = result[pred_name].as_continuous()
    resp = result[resp_name].as_continuous()
    respstd = np.nanstd(resp)
    squared_errors = (pred-resp)**2
    mse = np.sqrt(np.nanmean(squared_errors))
    return mse / respstd


def nmse_shrink(result, pred_name='pred', resp_name='resp', shrink=0.25):
    '''
    Same as MSE, but normalized by the std of the resp.
    Because it is more computationally expensive than MSE but is otherwise
    equivalent, we suggest using the MSE for fitting and use this as a
    post-fit performance metric only.
    '''

    X1 = result[pred_name].as_continuous()
    X2 = result[resp_name].as_continuous()

    keepidx = np.isfinite(X1) * np.isfinite(X2)
    if np.all(np.logical_not(keepidx)):
        log.debug("All values were NaN or inf in pred and resp")
        return 1

    X1 = X1[keepidx]
    X2 = X2[keepidx]

    bounds = np.round(np.linspace(0, len(X1) + 1, 11)).astype(int)
    E = np.zeros([10, 1])
    for ii in range(0, 10):
        if bounds[ii] == bounds[ii + 1]:
            log.info('No data in range?')

        P = np.mean(np.square(X2[bounds[ii]:bounds[ii + 1]]))

        if P > 0:
            E[ii] = np.sqrt(np.mean(np.square(X1[bounds[ii]:bounds[ii + 1]] -
                                X2[bounds[ii]:bounds[ii + 1]])) / P)
        else:
            E[ii] = 1

    mE = E.mean()
    sE = E.std()

    if mE < 1:
        # apply shrinkage filter to 1-E with factors self.shrink
        mse = 1 - nems.utils.shrinkage(1 - mE, sE, shrink)
    else:
        mse = mE

    return mse
