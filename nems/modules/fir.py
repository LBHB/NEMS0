from itertools import chain, repeat

import numpy as np
import scipy.signal
from scipy import interpolate

def get_zi(b, x):
    # This is the approach NARF uses. If the initial value of x[0] is 1,
    # this is identical to the NEMS approach. We need to provide zi to
    # lfilter to force it to return the final coefficients of the dummy
    # filter operation.
    n_taps = len(b)
    null_data = np.full(n_taps*2, x[0])
    zi = np.ones(n_taps-1)
    return scipy.signal.lfilter(b, [1], null_data, zi=zi)[1]


def _insert_zeros(coefficients, rate=1):
    if rate<=1:
        return coefficients

    d1 = int(np.ceil((rate-1)/2))
    d0 = int(rate-1-d1)
    s = coefficients.shape
    new_c = np.concatenate((np.zeros((s[0],s[1],d0)),
                            np.expand_dims(coefficients, axis=2),
                            np.zeros((s[0],s[1],d1))), axis=2)
    new_c = np.reshape(new_c, (s[0],s[1]*rate))
    return new_c


def per_channel(x, coefficients, bank_count=1, rate=1):
    '''Private function used by fir_filter().

    Parameters
    ----------
    x : array (n_channels, n_times) or (n_channels * bank_count, n_times)
        Input data. Can be sized two different ways:
        option 1: number of input channels is same as total channels in the
          filterbank, allowing a different stimulus into each filter
        option 2: number of input channels is same as number of coefficients
          in each fir filter, so that the same stimulus goes into each
          filter
    coefficients : array (n_channels * bank_count, n_taps)
        Filter coefficients. For ``x`` option 2, input channels are nested in
        output channel, i.e., filter ``filter_i`` of bank ``bank_i`` is at
        ``coefficients[filter_i * n_banks + bank_i]``.
    bank_count : int
        Number of filters in each bank.

    Returns
    -------
    signal : array (bank_count, n_times)
        Filtered signal.
    '''
    # Make sure the number of input channels (x) match the number FIR filters
    # provided (we have a separate filter for each channel). The `zip` function
    # doesn't require the iterables to be the same length.
    n_in = len(x)
    if rate > 1:
        coefficients = _insert_zeros(coefficients, rate)
        print(coefficients)
    n_filters = len(coefficients)
    n_banks = int(n_filters / bank_count)
    if n_filters == n_in:
        # option 1: number of input channels is same as total channels in the
        # filterbank, allowing a different stimulus into each filter
        all_x = iter(x)
    elif n_filters == n_in * bank_count:
        # option 2: number of input channels is same as number of coefficients
        # in each fir filter, so that the same stimulus goes into each
        # filter
        one_x = tuple(x)
        all_x = chain.from_iterable([one_x for _ in range(bank_count)])
    else:
        if bank_count == 1:
            desc = '%i FIR filters' % n_filters
        else:
            desc = '%i FIR filter banks' % n_banks
        raise ValueError(
            'Dimension mismatch. %s channels provided for %s.' % (n_in, desc))

    c_iter = iter(coefficients)
    out = np.zeros((bank_count, x.shape[1]))
    for i_out in range(bank_count):
        for i_bank in range(n_banks):
            x_ = next(all_x)
            c = next(c_iter)
            # It is slightly more "correct" to use lfilter than convolve at
            # edges, but but also about 25% slower (Measured on Intel Python
            # Dist, using i5-4300M)
            zi = get_zi(c, x_)
            r, zf = scipy.signal.lfilter(c, [1], x_, zi=zi)
            out[i_out] += r
    return out


def basic(rec, i='pred', o='pred', coefficients=[], rate=1):
    """
    apply fir filters of the same size in parallel. convolve in time, then
    sum across channels

    coefficients : 2d array
        all coefficients matrix shape=channel X time lag, for which
        .shape[0] matched to the channel count of the input

    input :
        nems signal named in 'i'. must have dimensionality matched to size
        of coefficients matrix.
    output :
        nems signal in 'o' will be 1 x time singal (single channel)
    """

    fn = lambda x: per_channel(x, coefficients, rate=1)
    return [rec[i].transform(fn, o)]


def pz_coefficients(poles=None, zeros=None, delays=None,
                    gains=None, n_coefs=10, fs=100):
    """
    helper funciton to generate coefficient matrix.
    """
    n_filters = len(gains)
    coefficients = np.zeros((n_filters, n_coefs))
    fs2 = 5*fs
    for j in range(n_filters):
        t = np.arange(0, n_coefs*5+1) / fs2
        h = scipy.signal.ZerosPolesGain(zeros[j], poles[j], gains[j], dt=1/fs2)
        tout, ir = scipy.signal.dimpulse(h, t=t)
        f = interpolate.interp1d(tout, ir[0][:,0], bounds_error=False,
                                 fill_value=0)

        tnew = np.arange(0,n_coefs)/fs - delays[j,0]/fs
        coefficients[j,:] = f(tnew)

    return coefficients


def pole_zero(rec, i='pred', o='pred', poles=None, zeros=None, delays=None,
              gains=None, n_coefs=10, rate=1):
    """
    apply pole_zero -defined filter
    generate impulse response and then call as if basic fir filter

    input :
        nems signal named in 'i'. must have dimensionality matched to size
        of coefficients matrix.
    output :
        nems signal in 'o' will be 1 x time singal (single channel)
    """

    coefficients = pz_coefficients(poles=poles, zeros=zeros, delays=delays,
                                   gains=gains, n_coefs=n_coefs, fs=rec[i].fs)

    fn = lambda x: per_channel(x, coefficients, rate=1)
    return [rec[i].transform(fn, o)]

def fir_dexp_coefficients(phi=None, n_coefs=20):
    """
    helper funciton to generate dexp coefficient matrix.
    """
    N_chans, N_parms = phi.shape

    if N_parms != 6:
        raise ValueError('FIR_DEXP needs exactly 6 parameters per channel')

    lat1=phi[:,0]
    tau1=np.abs(phi[:,1])
    A1=phi[:,2]
    lat2=phi[:,3]
    tau2=np.abs(phi[:,4])
    A2=phi[:,5]

    coefs = np.zeros((N_chans, n_coefs))

    t = np.arange(0,n_coefs)
    for c in range(N_chans):
        coefs[c,:]=A1[c]*(np.exp(-tau1[c]*(t-lat1[c])) -
                          np.exp(-tau1[c]*5*(t-lat1[c]))) * (t-lat1[c]>0) + \
                   A2[c]*(np.exp(-tau2[c]*(t-lat2[c])) -
                          np.exp(-tau2[c]*5*(t-lat2[c]))) * (t-lat2[c]>0)

    return coefs


def fir_dexp(rec, i='pred', o='pred', phi=None, n_coefs=10, rate=1):
    """
    apply pole_zero -defined filter
    generate impulse response and then call as if basic fir filter

    input :
        nems signal named in 'i'. must have dimensionality matched to size
        of coefficients matrix.
    output :
        nems signal in 'o' will be 1 x time singal (single channel)
    """

    coefficients = fir_dexp_coefficients(phi, n_coefs)

    fn = lambda x: per_channel(x, coefficients, rate=1)
    return [rec[i].transform(fn, o)]


def filter_bank(rec, i='pred', o='pred', coefficients=[], bank_count=1, rate=1):
    """
    apply multiple basic fir filters of the same size in parallel, producing
    one output channel per filter.

    bank_count : integer
        number of filters
    coefficients : 2d array
        all filters are stored in a single coefficients matrix, for which
        .shape[0] must be an integer multiple of bank_count.

    input :
        nems signal named in 'i'. must have dimensionality matched to size
        of coefficients matrix. if you'd like to apply each filter to the
        same inputs, you should process the 'i' signal fiter using the
        signal_mod.replicate_channels module
    output :
        nems signal in 'o' will be bank_count x time matrix

    TODO: test, optimize. maybe structure coefficients more logically?
    """

    fn = lambda x: per_channel(x, coefficients, bank_count, rate=rate)
    return [rec[i].transform(fn, o)]


def fir_exp_coefficients(tau, a=1, b=0, n_coefs=15):
    '''
    Generate coefficient matrix using a three-parameter exponential equation.
    y = a*e^(-x/tau) + b
    '''
    n_chans = tau.shape[0]
    coefs = np.zeros((n_chans, n_coefs))
    t = np.arange(0, n_coefs)
    for c in range(n_chans):
        coefs[c, :] = a[c]*np.exp(-t/tau[c]) + b[c]

    return coefs


def fir_exp(rec, tau, a=1, b=0, i='pred', o='pred', n_coefs=15, rate=1):
    '''
    Calculate coefficients based on three-parameter exponential equation
    y = a*e^(-x/tau) + b,
    then call as basic fir.

    '''
    coefficients = fir_exp_coefficients(tau, a, b, n_coefs)
    fn = lambda x: per_channel(x, coefficients, rate=rate)
    return [rec[i].transform(fn, o)]
