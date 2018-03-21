import pytest
import numpy as np


# ------------------------------------------------------------------------------
# Weight channels
# ------------------------------------------------------------------------------
from nems.modules import weight_channels as wc


def test_gaussian_coefficients():
    mean = [0.3, 0.6]
    sd = [0.25, 0.25]
    n_chan_in = 4
    expected = np.array(
        [[ 0.20357697,  0.40995367,  0.30370115,  0.08276822],
         [ 0.02563427,  0.17138805,  0.42154657,  0.38143111]]
    )

    coefs = wc.gaussian_coefficients(mean, sd, n_chan_in)
    assert coefs.shape == (2, 4)

    np.testing.assert_allclose(coefs.sum(axis=1), 1)
    np.testing.assert_allclose(expected, coefs, rtol=1e-5)
