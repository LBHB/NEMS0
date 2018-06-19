import pytest


from nems.initializers import from_keywords
from nems.priors import set_mean_phi


@pytest.fixture
def simple_modelspec():
    return from_keywords('wcg18x2_fir2x15_dexp1')


@pytest.fixture
def simple_modelspec_with_phi(simple_modelspec):
    ms = from_keywords('wcg18x2_fir2x15_dexp1')
    return set_mean_phi(ms)
