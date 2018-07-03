import pytest


from nems.initializers import from_keywords
from nems.priors import set_mean_phi


@pytest.fixture
def simple_modelspec():
    return from_keywords('wc.18x2.g-fir.2x15-dexp.1')


@pytest.fixture
def simple_modelspec_with_phi(simple_modelspec):
    ms = from_keywords('wc.18x2.g-fir.2x15-dexp.1')
    return set_mean_phi(ms)
