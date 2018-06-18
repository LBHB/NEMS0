import numpy as np

from nems.priors import set_percentile_bounds


def test_set_percentile_bounds(simple_modelspec):
    modelspec = set_percentile_bounds(simple_modelspec, 0, 1)
    c = modelspec[1]['bounds']['coefficients']
    assert np.all(np.equal(c[0], -np.inf))
    assert np.all(np.equal(c[1], np.inf))

    modelspec = set_percentile_bounds(simple_modelspec, 0.1, 0.9)
    c = modelspec[1]['bounds']['coefficients']
    assert np.all(np.allclose(c[0], -1.28155157))
    assert np.all(np.allclose(c[1], 1.28155157))
