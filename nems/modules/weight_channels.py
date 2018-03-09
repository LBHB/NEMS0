import numpy as np

def weight_channels(rec, i, o, coefficients):
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
    fn = lambda x: coefficients @ x
    return [rec[i].transform(fn, o)]

def _gaussian_coefs(i, o, mn, sig, num_chan_in):
    '''
    helper function for gaussian
    '''
    num_chan_out=len(mn)
    coefficients = np.zeros([num_chan_out, num_chan_in])
    for k in range(0, num_chan_out):
        m = mn[k] * num_chan_in
        s = sig[k] * num_chan_in

        x = np.arange(0, num_chan_in)
        coefficients[k, :] = np.exp(-np.square((x - m) / s))
        coefficients[k, :] = coefficients[k, :] / np.sum(coefficients[k, :])
    
    return coefficients

def gaussian(rec, i, o, mn, sig, num_chan_in):
    '''
    Parameters
    ----------
    mn,sig = (mean,std) of num_chan_out gaussian functions
    num_chan_in
    num_chan_out = len(mn)
    generates num_chan_out X num_chan_in coefficients matrix that's fed into
    standard weight_channels
    '''
    coefficients = _gaussian_coefs(i, o, mn, sig, num_chan_in)
    
    fn = lambda x: coefficients @ x

    return [rec[i].transform(fn, o)]


