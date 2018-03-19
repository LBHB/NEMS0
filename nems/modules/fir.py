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


def per_channel(x, coefficients):
    '''
    Private function used by fir_filter().
    '''
    result = []

    # Make sure the number of input channels (x) match the number FIR filters
    # provided (we have a separate filter for each channel). The `zip` function
    # doesn't require the iterables to be the same length.
    if len(x) != len(coefficients):
        m = 'Dimension mismatch. Number of channels and filters must match. ' \
            '{} channels provided for {} FIR filters.'
        raise ValueError(m.format(len(x), len(coefficients)))

    for x, c in zip(x, coefficients):
        # It is slightly more "correct" to use lfilter than convolve at edges, but
        # but also about 25% slower (Measured on Intel Python Dist, using i5-4300M)
        zi = get_zi(c, x)
        r, zf = scipy.signal.lfilter(c, [1], x, zi=zi)
        # TODO: Use convolve. Why is this giving the wrong answer?
        # r = np.convolve(c, x, mode='same')
        result.append(r[np.newaxis])
    result = np.concatenate(result)
    return np.sum(result, axis=-2, keepdims=True)


def basic(rec, i, o, coefficients):
    fn = lambda x: per_channel(x, coefficients)
    return [rec[i].transform(fn, o)]
