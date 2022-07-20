from itertools import chain, repeat

import numpy as np
import scipy.signal
from scipy import interpolate
from scipy.ndimage.filters import convolve1d

def get_zi(b, x):
    # This is the approach NARF uses. If the initial value of x[0] is 1,
    # this is identical to the NEMS approach. We need to provide zi to
    # lfilter to force it to return the final coefficients of the dummy
    # filter operation.
    n_taps = len(b)
    #null_data = np.full(n_taps*2, x[0])
    null_data = np.full(n_taps*2, 0)
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


def per_channel(x, coefficients, bank_count=1, non_causal=0, rate=1,
                cross_channels=False):
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
    if bank_count>0:
        n_banks = int(n_filters / bank_count)
    else:
        n_banks = n_filters
    if cross_channels:
        # option 0: user has specified that each filter should be applied to
        # each input channel (requires bank_count==1)
        # TODO : integrate with core loop below instead of pasted hack
        out = np.zeros((n_in*n_filters, x.shape[1]))
        i_out=0
        for i_in in range(n_in):
            x_ = x[i_in]
            for i_bank in range(n_filters):
                c = coefficients[i_bank]
                zi = get_zi(c, x_)
                r, zf = scipy.signal.lfilter(c, [1], x_, zi=zi)
                out[i_out] = r
                i_out+=1
        return out
    elif n_filters == n_in:
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
            if non_causal:
                # reverse model (using future values of input to predict)
                x_ = np.roll(x_, -non_causal)

            # It is slightly more "correct" to use lfilter than convolve at
            # edges, but but also about 25% slower (Measured on Intel Python
            # Dist, using i5-4300M)
            zi = get_zi(c, x_)
            r, zf = scipy.signal.lfilter(c, [1], x_, zi=zi)
            out[i_out] += r
    return out

def fir_conv2(x, coefficients, bank_count=1, non_causal=0, rate=1):
    '''
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
    if bank_count>0:
        n_banks = int(n_filters / bank_count)
    else:
        n_banks = n_filters
    if cross_channels:
        # option 0: user has specified that each filter should be applied to
        # each input channel (requires bank_count==1)
        # TODO : integrate with core loop below instead of pasted hack
        out = np.zeros((n_in*n_filters, x.shape[1]))
        i_out=0
        for i_in in range(n_in):
            x_ = x[i_in]
            for i_bank in range(n_filters):
                c = coefficients[i_bank]
                zi = get_zi(c, x_)
                r, zf = scipy.signal.lfilter(c, [1], x_, zi=zi)
                out[i_out] = r
                i_out+=1
        return out
    elif n_filters == n_in:
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
            if non_causal:
                # reverse model (using future values of input to predict)
                #x_ = np.roll(x_, -(len(c) - 1))
                x_ = np.roll(x_, -non_causal)

            # It is slightly more "correct" to use lfilter than convolve at
            # edges, but but also about 25% slower (Measured on Intel Python
            # Dist, using i5-4300M)
            zi = get_zi(c, x_)
            r, zf = scipy.signal.lfilter(c, [1], x_, zi=zi)
            out[i_out] += r
    return out


def basic(rec, i='pred', o='pred', non_causal=0, coefficients=[], rate=1,
          offsets=0, **kwargs):
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
    offset : float
        Number of milliseconds to offset the coefficients by

    """

    if not np.all(offsets == 0):
        fs = rec[i].fs
        coefficients = _offset_coefficients(coefficients, offsets, fs)
    fn = lambda x: per_channel(x, coefficients, non_causal=non_causal,
                               rate=1)

    return [rec[i].transform(fn, o)]


def _offset_coefficients(coefficients, offsets, fs, pad_bins=False):
    '''
    Compute new coefficients with the same shape that are offset by some time.

    Parameters:
    ----------
    coefficients : 2d ndarray
        FIR coefficients to use as a starting point.
    offsets : 2d ndarray
        Amount of offset (in milliseconds) to apply to each channel.
    fs : int
        Sampling rate of the recording that the FIR will be applied to.
        Used to determine bin size.
    pad_bins : Boolean
        If true, add bins to the end of coefficients (for positive offsets)
        or the start of coefficients (for negative offsets) instead of
        clipping bins.

    Returns:
    -------
    new_coeffs : 2d ndarray
        Time-shifted FIR coefficients. Some non-zero coefficients near the
        first and last coefficients may have been clipped. Offsets that are
        not integer multiples of the bin size will result in wider "peaks,"
        but an approximately equal area under the curve.

    '''
    if pad_bins:
        coefficients = coefficients.copy()
        max_offset = np.max(offsets)
        bins_to_pad = int(np.ceil(np.abs(max_offset)*fs/1000))
        d1, d2 = coefficients.shape
        empty_bins = np.zeros((d1, bins_to_pad))
        if max_offset >= 0:
            coefficients = np.concatenate((coefficients, empty_bins), axis=1)
        else:
            coefficients = np.concatenate((empty_bins, coefficients), axis=1)

    new_coeffs = np.empty_like(coefficients, dtype=np.float64)
    for k, offset in enumerate(offsets):
        mixed_bin = np.abs(offset)*fs/1000
        whole_bin = int(np.floor(mixed_bin))
        frac_bin = np.remainder(mixed_bin, 1)
        offset_coefficients = []

        kernel = np.zeros((3+(whole_bin)*2))
        if offset >= 0:
            kernel[-1] = frac_bin
            kernel[-2] = 1-frac_bin
        else:
            kernel[0] = frac_bin
            kernel[1] = 1-frac_bin

        offset_coefficients = convolve1d(coefficients[k], kernel,
                                     mode='constant')
        new_coeffs[k] = offset_coefficients

    return new_coeffs


def offset_area_check(original_coefficients, offsets_range=(-10,30),
                      n_samples=80, fs=100):
    '''
    Use to (roughly) check that offset algorithm isn't changing total gain.

    If the offset range causes non-zero coefficients to be clipped, there will
    obviously be some noticeable changes to the area under the curve. Otherwise,
    the AUC difference vs offset amount relationship should be more or less
    flat and very close to zero.

    Parameters:
    ----------
    original_coefficients : 2d ndarray
        FIR coefficients to test against.
        Ex: np.array([[0, 1, 0, 0, 0, 0]])
    offsets_range : 2-tuple
        Specifies the minimum and maximum offsets to test, in milliseconds.
    n_samples : int
        Number of offset values to test, evenly spaced within offsets_range.
    fs : int
        Sampling rate to assume for the coefficients.

    Returns:
    -------
    offsets, diffs : 1d ndarrays
        Each pair represents the ms offset used to shift coefficients and
        the difference in the area under the curve for the original vs the
        shifted coefficients.

    '''
    offsets = np.linspace(*offsets_range, n_samples)
    if original_coefficients.shape[0] > 1:
        original_coefficients = np.array([original_coefficients[0]])
    diffs = []
    for o in offsets:
        off = np.array([o])
        c = _offset_coefficients(original_coefficients, off, fs)
        a1 = np.trapz(original_coefficients)
        a2 = np.trapz(c)
        diffs.append(a1 - a2)
    diffs = np.array(diffs).flatten()

    return offsets, diffs


def pz_coefficients(poles=None, zeros=None, delays=None,
                    gains=None, n_coefs=10, fs=100, **kwargs):
    """
    helper funciton to generate coefficient matrix.
    """
    n_filters = len(gains)
    coefficients = np.zeros((n_filters, n_coefs))
    fs2 = 5*fs
    #poles = (poles.copy()+1) % 2 - 1
    #zeros = (zeros.copy()+1) % 2 - 1
    poles = poles.copy()
    poles[poles>1]=1
    poles[poles<-1]=-1
    zeros = zeros.copy()
    zeros[zeros>1]=1
    zeros[zeros<-1]=-1
    for j in range(n_filters):
        t = np.arange(0, n_coefs*5+1) / fs2
        h = scipy.signal.ZerosPolesGain(zeros[j], poles[j], gains[j], dt=1/fs2)
        tout, ir = scipy.signal.dimpulse(h, t=t)
        f = interpolate.interp1d(tout, ir[0][:,0], bounds_error=False,
                                 fill_value=0)

        tnew = np.arange(0, n_coefs)/fs - delays[j,0] + 1/fs
        coefficients[j,:] = f(tnew)

    return coefficients


def pole_zero(rec, i='pred', o='pred', poles=None, zeros=None, delays=None,
              gains=None, n_coefs=10):
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


def do_coefficients(f1s=1, taus=0.5, delays=1, gains=1, n_coefs=10, **kwargs):
    """
    generate fir filter from damped oscillator coefficients
    :param f1s:
    :param taus:
    :param delays:
    :param gains:
    :param n_coefs:
    :param kwargs:  padding for extra stuff if implemented in a framework with
                    generic coef-generating functions
    :return:
    """
    t = np.arange(n_coefs) - delays
    coefficients = np.sin(f1s * t) * np.exp(-taus * t) * gains
    coefficients[t<0] = 0

    return coefficients

def da_coefficients(**kwargs):
    """
    backwards compatible, alias for more smartly-named do_coefficients
    :param kwargs:
    :return:
    """

    return do_coefficients(**kwargs)

def damped_oscillator(rec, i='pred', o='pred', f1s=1, taus=0.5, delays=1,
                      gains=1, n_coefs=10, bank_count=1, cross_channels=False, **kwargs):
    """
    apply damped oscillator-defined filter
    generate impulse response and then call as if basic fir filter

    input :
        nems signal named in 'i'. must have dimensionality matched to size
        of coefficients matrix.
    output :
        nems signal in 'o' will be 1 x time signal (single channel)
    """

    coefficients = do_coefficients(f1s=f1s, taus=taus, delays=delays,
                                   gains=gains, n_coefs=n_coefs)

    fn = lambda x: per_channel(x, coefficients, bank_count=bank_count,
                               cross_channels=cross_channels, rate=1)
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


def filter_bank(rec, i='pred', o='pred', non_causal=0, coefficients=[],
                bank_count=1, rate=1, cross_channels=False, **kwargs):
    """
    apply multiple basic fir filters of the same size in parallel, producing
    one output channel per filter.

    bank_count : integer
        number of filters
    coefficients : 2d array
        all filters are stored in a single coefficients matrix, for which
        .shape[0] must be an integer multiple of bank_count.
    cross_channels : (False) if True, apply each 1d filter to each input channel,
        producing bank_count * x.shape[0] output channels
        (requires bank_count==1)
    input :
        nems signal named in 'i'. must have dimensionality matched to size
        of coefficients matrix. if you'd like to apply each filter to the
        same inputs, you should process the 'i' signal fiter using the
        signal_mod.replicate_channels module
    output :
        nems signal in 'o' will be bank_count x time matrix

    TODO: test, optimize. maybe structure coefficients more logically?
    TODO: filterbanks all handled properly?
    """
    fn = lambda x: per_channel(x, coefficients, bank_count,
                               non_causal=non_causal, rate=rate,
                               cross_channels=cross_channels)
    return [rec[i].transform(fn, o)]


def fir_exp_coefficients(tau=1, a=1, b=0, s=0, n_coefs=15):
    '''
    Generate coefficient matrix using a four-parameter exponential equation.
    y = a*e^(-(x-s)/tau) + b
    '''
    t = np.arange(n_coefs)
    coefs = a*np.exp(-(t-s)/tau) + b

    return coefs


def fir_exp(rec, tau, a=1, b=0, s=0, i='pred', o='pred', n_coefs=15, rate=1):
    '''
    Calculate coefficients based on three-parameter exponential equation
    y = a*e^(-x/tau) + b,
    then call as basic fir.

    '''
    coefficients = fir_exp_coefficients(tau, a, b, n_coefs)
    fn = lambda x: per_channel(x, coefficients, rate=rate)
    return [rec[i].transform(fn, o)]
