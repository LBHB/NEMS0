import pytest

import numpy as np

from nems.fitters.mappers import simple_vector, to_bounds_array


def test_simple_vector_subset(simple_modelspec_with_phi):
    packer, unpacker, _ = simple_vector(simple_modelspec_with_phi)
    phi = packer(simple_modelspec_with_phi)
    assert len(phi) == 38

    subset = [0, 2]
    packer, unpacker = simple_vector(simple_modelspec_with_phi, subset)
    phi = packer(simple_modelspec_with_phi)
    assert len(phi) == 8

    vector = np.arange(8)
    new_modelspec = unpacker(vector)

    assert new_modelspec[0]['phi']['mean'].tolist() == [0, 1]
    assert new_modelspec[0]['phi']['sd'].tolist() == [2, 3]
    dexp_expected = {
        'amplitude': 4,
        'base': 5,
        'kappa': 6,
        'shift': 7
    }
    assert new_modelspec[2]['phi'] == dexp_expected

    assert new_modelspec[1]['phi']['coefficients'].tolist() == \
        new_modelspec[1]['phi']['coefficients'].tolist()

    # Check that identity is the same
    assert id(new_modelspec[1]['phi']['coefficients']) == \
        id(new_modelspec[1]['phi']['coefficients'])
    assert id(new_modelspec[0]['phi']) == id(new_modelspec[0]['phi'])
    assert id(simple_modelspec_with_phi) == id(new_modelspec)


def test_simple_vector_bounds_subset(simple_modelspec_with_phi):
    packer, unpacker, bounds = simple_vector(simple_modelspec_with_phi)
    phi = packer(simple_modelspec_with_phi)
    lb, ub = bounds(simple_modelspec_with_phi)
    assert np.all(np.equal(lb, -np.inf))
    assert np.all(np.equal(ub, np.inf))
    assert len(lb) == 38
    assert len(ub) == len(lb)
    assert len(ub) == len(phi)

    subset = [0, 2]
    packer, unpacker, bounds = simple_vector(simple_modelspec_with_phi, subset)
    phi = packer(simple_modelspec_with_phi)
    lb, ub = bounds(simple_modelspec_with_phi)
    phi = np.array(phi)
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(np.equal(lb, -np.inf))
    assert np.all(np.equal(ub, np.inf))
    assert len(lb) == 8
    assert len(ub) == len(lb)
    assert len(ub) == len(phi)

    simple_modelspec_with_phi[0].setdefault('bounds', {})['mean'] = (None, 1)
    lb, ub = bounds(simple_modelspec_with_phi)
    assert np.all(np.equal(lb, -np.inf))
    assert np.all(np.equal(ub[:2], 1))
    assert np.all(np.equal(ub[2:], np.inf))

    simple_modelspec_with_phi[0].setdefault('bounds', {})['mean'] = (-1, [2, 3])
    lb, ub = bounds(simple_modelspec_with_phi)
    assert np.all(np.equal(lb[:2], -1))
    assert np.all(np.equal(lb[2:], -np.inf))
    assert np.all(np.equal(ub[:2], [2, 3]))
    assert np.all(np.equal(ub[2:], np.inf))


def test_benchmark_packer_unpacker(benchmark, simple_modelspec_with_phi):
    packer, unpacker, _ = simple_vector(simple_modelspec_with_phi)
    vector = np.arange(38)
    benchmark(unpacker, vector)


def test_to_bounds_array():
    phi = np.arange(12).reshape((2, 6))

    value = to_bounds_array((None, None), phi, 'lower')
    assert np.all(value == -np.inf)

    value = to_bounds_array((None, None), phi, 'upper')
    assert np.all(value == np.inf)

    value = to_bounds_array((None, -5), phi, 'lower')
    assert np.all(value == -np.inf)

    value = to_bounds_array((-5, None), phi, 'lower')
    assert np.all(value == -5)

    value = to_bounds_array((None, -5), phi, 'upper')
    assert np.all(value == -5)

    value = to_bounds_array((-5, None), phi, 'upper')
    assert np.all(value == np.inf)

    value = to_bounds_array((-5, 5), phi, 'upper')
    assert np.all(value == 5)

    upper_bound = np.ones_like(phi)
    value = to_bounds_array((-5, upper_bound), phi, 'upper')
    assert np.all(value == value)

    upper_bound = np.ones_like(phi)
    value = to_bounds_array((-5, upper_bound), phi, 'upper')
    assert np.all(value == value)


################################################################################
# Copied from test_bounds.py. Originally created by @jacob and updated to work
# with new bounds code.
################################################################################
@pytest.fixture()
def bounds_modelspec():
    modelspec = [
            {"fn": "nems.modules.weight_channels.gaussian",
             "fn_kwargs": {"i": "stim", "o": "pred", "n_chan_in": 18},
             "phi": {"mean": np.array([-0.024397602626293702,
                                       0.4698089169449159]),
                     "sd": np.array([0.8668943517628687,
                                     0.36811997663567037])}},

            {"fn": "nems.modules.nonlinearity.double_exponential",
             "fn_kwargs": {"i": "pred", "o": "pred"},
             "phi": {"amplitude": 2.071742144205505,
                     "base": -0.3911174879125417,
                     "kappa": 0.4119910862820081,
                     "shift": 0.34811284718401403}}
            ]
    return modelspec


def test_scalar_bounds(bounds_modelspec):
    bounds_modelspec[0]['bounds'] = {
        'mean': (None, [1.1, 1.2]),
        'sd': ([0.0, 0.7], 6)
    }
    bounds_modelspec[1]['bounds'] = {
        'kappa': (-1, None),
        'base': (None, None),
        'amplitude': (0, 7),
        'shift': (None, 5)
    }

    packer, unpacker, bounds = simple_vector(bounds_modelspec)
    lb, ub = bounds(bounds_modelspec)
    lb_expected = [-np.inf, -np.inf, 0, 0.7, 0, -np.inf, -1, -np.inf]
    ub_expected = [1.1, 1.2, 6, 6, 7, np.inf, np.inf, 5]
    assert np.all(np.equal(lb, lb_expected))
    assert np.all(np.equal(ub, ub_expected))


def test_partial_definition(modelspec):
    modelspec[0]['bounds'] = {
        'mean': (None, 10)
    }
    # Don't need to assert anything here, just shouldn't get an error
    # for leaving 'sd' bounds undefined.
    x = bounds_vector([modelspec[0]])
