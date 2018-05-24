import pytest
import numpy as np

import nems.recording as recording
import nems.signal as signal
import nems.modules.stp as stp
import nems.modules.fir as fir
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Weight channels
# ------------------------------------------------------------------------------
from nems.modules import weight_channels as wc


def test_gaussian_coefficients():
    mean = [0.3, 0.6]
    sd = [0.25, 0.25]
    n_chan_in = 4
    expected = np.array(
        [[0.20357697,  0.40995367,  0.30370115,  0.08276822],
         [0.02563427,  0.17138805,  0.42154657,  0.38143111]]
    )

    coefs = wc.gaussian_coefficients(mean, sd, n_chan_in)
    assert coefs.shape == (2, 4)

    np.testing.assert_allclose(coefs.sum(axis=1), 1)
    np.testing.assert_allclose(expected, coefs, rtol=1e-5)


def test_stp():

    nchans = 1
    fs = 100
    data = np.concatenate([np.zeros([1, 10]), np.ones([1, 20]),
                          np.zeros([1, 20]), np.ones([1, 5]), np.zeros([1, 5]),
                          np.ones([1, 5]), np.zeros([1, 5]),
                          np.ones([1, 5]), np.zeros([1, 10])], axis=1)

    kwargs = {
        'data': data,
        'name': 'pred',
        'recording': 'rec',
        'chans': ['chan' + str(n) for n in range(nchans)],
        'fs': fs,
        'meta': {
            'for_testing': True,
            'date': "2018-01-10",
            'animal': "Donkey Hotey",
            'windmills': 'tilting'
        },
    }
    pred = signal.RasterizedSignal(**kwargs)
    rec = recording.Recording({'pred': pred})

    u = np.array([5.0])
    tau = np.array([8.0])

    r = stp.short_term_plasticity(rec, 'pred', 'pred_out', u=u, tau=tau)
    pred_out = r[0]

    plt.figure()
    plt.plot(pred.as_continuous().T)
    plt.plot(pred_out.as_continuous().T)
    # print(pred_out.as_continuous().T)
    # Y = stp._stp(X, u, tau)


def test_firbank():
    x = np.random.randn(2, 100)
    coefficients = np.zeros([4, 10])
    coefficients[0, 1] = 1
    coefficients[2, 3] = -0.5

    bank_count = 2

    # test one stimulus, multi-banks
    y = fir.per_channel(x, coefficients, bank_count)

    assert(y.shape == (2, 100))
    assert(np.sum(np.abs(y[0, :98] + 2 * y[1, 2:])) == 0)

    x2 = np.concatenate((x, -x), axis=0)
    y2 = fir.per_channel(x2, coefficients, bank_count)

    assert(y2.shape == (2, 100))
    assert(np.sum(np.abs(y2[0, :98] - 2 * y2[1, 2:])) == 0)
