import pytest
import numpy as np

from nems.fitters.mappers import bounds_vector, simple_vector

# NOTE: This only tests that bounds definitions in the modelspec are
#       correctly converted to flattened lists of tuples for use by
#       scipy_minimize and coordinate_descent. These tests
#       *do not* make any guarantees about the behavior of the fitters.


@pytest.fixture()
def modelspec():
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


def test_all_blank(modelspec):
    do_phi, undo_phi = simple_vector(modelspec)
    phi = do_phi(modelspec)
    bounds = bounds_vector(modelspec)
    assert [b == (None, None) for b in bounds]
    assert len(bounds) == len(phi)


def test_scalar_bounds(modelspec):
    modelspec[1]['bounds'] = {
            'amplitude': (0, 5),
            'base': (None, None),
            'kappa': (-1, None),
            'shift': (None, 5)
            }
    assert bounds_vector([modelspec[1]]) == [
            (0, 5), (None, None), (-1, None), (None, 5)
            ]


def test_mixed_bounds(modelspec):
    modelspec[0]['bounds'] = {
            'mean': (None, [1.1, 1.2]),
            'sd': ([0.0, 0.7], 5)
            }
    assert bounds_vector([modelspec[0]]) == [
            (None, 1.1), (None, 1.2), (0.0, 5), (0.7, 5)
            ]


def test_partial_definition(modelspec):
    modelspec[0]['bounds'] = {
            'mean': (None, 10)
            }
    # Don't need to assert anything here, just shouldn't get an error
    # for leaving 'sd' bounds undefined.
    x = bounds_vector([modelspec[0]])


def test_wrong_shape(modelspec):
    modelspec[0]['bounds'] = {
            'mean': ([1,1,1,1,1,1,11,1,1,1,1,11,1,1,1,], None),
            }
    with pytest.raises(ValueError):
        x = bounds_vector([modelspec[0]])
