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
            {"fn": "nems.modules.weight_channels.gaussian",                 # 0
             "fn_kwargs": {"i": "stim", "o": "pred", "n_chan_in": 18},
             "phi": {"mean": np.array([-0.024397602626293702,
                                       0.4698089169449159]),
                     "sd": np.array([0.8668943517628687,
                                     0.36811997663567037])}},
            {"fn": "nems.modules.fir.basic",                                # 1
             "fn_kwargs": {"i": "pred", "o": "pred"},
             "phi": {"coefficients": np.array(
                     [[0.10502974890411182, 0.20095323302837909,
                       0.042312432570341646, -0.06833534626062727,
                       -0.036781123769710954, -0.0027752778626991383,
                       0.012874046657580506, 0.0033524582143592612,
                       -0.028675050843844443, -0.015909507248576007,
                       -0.0030860129510280605, -0.021163107450481472,
                       -0.00706960597651072, 0.004145417470190893,
                       -0.021681201525654416],
                      [-0.08689982551226674, -0.15406586584259588,
                       -0.01836452127636852, 0.06258829988549444,
                       0.03914873497869759, -0.005259606682386534,
                       -0.01705566679402844, -0.009044941264732967,
                       0.023692711459835307, 0.009972787486504417,
                       -0.0009437290855890642, 0.012419765394016296,
                       0.0056564231348495446, -0.003235210676296292,
                       0.011601042425136987]])}},
            {"fn": "nems.modules.levelshift.levelshift",                    # 2
             "fn_kwargs": {"i": "pred", "o": "pred"},
             "phi": {"level": np.array([0.15268252732814436])}},
            {"fn": "nems.modules.nonlinearity.double_exponential",          # 3
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
    modelspec[3]['bounds'] = {
            'amplitude': (0, 5),
            'base': (None, None),
            'kappa': (-1, None),
            'shift': (None, 5)
            }
    assert bounds_vector([modelspec[3]]) == [
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
