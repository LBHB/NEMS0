import numpy as np
import scipy.signal


def get_zi(b, x):
    # This is the approach NARF uses. If the initial value of x[0] is 1,
    # this is identical to the NEMS approach. We need to provide zi to
    # lfilter to force it to return the final coefficients of the dummy
    # filter operation.
    n_taps = len(b)
    null_data = np.full(n_taps*2, x[0])
    zi = np.ones(n_taps-1)
    return scipy.signal.lfilter(b, [1], null_data, zi=zi)[1]


def per_channel(x, coefficients, bank_count=1):
    '''Private function used by fir_filter().

    Parameters
    ----------
    x : array (n_channels, n_times)
        Input data.
    coefficients : array (n_channels * bank_count, n_taps)
        Filter coefficients.
    bank_count : int
        Number of filters in each bank.

    Returns
    -------
    signal : array (n_channels * bank_count, n_times)
        Filtered signal.
    '''
    # Make sure the number of input channels (x) match the number FIR filters
    # provided (we have a separate filter for each channel). The `zip` function
    # doesn't require the iterables to be the same length.
    n_in = len(x)
    n_out = n_in * bank_count
    if len(coefficients) != n_out:
        if bank_count == 1:
            desc = '%i FIR filters' % len(coefficients)
        else:
            n_banks = len(coefficients) / bank_count
            desc = '%i FIR filter banks' % n_banks
        raise ValueError(
            'Dimension mismatch. %s channels provided for %s.' % (n_in, desc))

    out = np.empty((n_out, x.shape[1]))
    for i_in in range(n_in):
        x_ = x[i_in]
        for i_bank in range(0, n_out, n_in):
            # It is slightly more "correct" to use lfilter than convolve at
            # edges, but but also about 25% slower (Measured on Intel Python
            # Dist, using i5-4300M)
            c = coefficients[i_bank + i_in]
            zi = get_zi(c, x_)
            r, zf = scipy.signal.lfilter(c, [1], x_, zi=zi)
            # TODO: Use convolve. Why is this giving the wrong answer?
            # r = np.convolve(c, x, mode='same')
            out[i_bank + i_in] = r
    return out


def basic(rec, i='pred', o='pred', coefficients=[]):
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

    fn = lambda x: per_channel(x, coefficients)
    return [rec[i].transform(fn, o)]


def filter_bank(rec, i='pred', o='pred', coefficients=[], bank_count=1):
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

    fn = lambda x: per_channel(x, coefficients, bank_count)
    return [rec[i].transform(fn, o)]
