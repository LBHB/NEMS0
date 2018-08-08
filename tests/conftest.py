import pytest

import numpy as np

from nems.recording import Recording
from nems.initializers import from_keywords
from nems.priors import set_mean_phi


@pytest.fixture
def simple_modelspec():
    return from_keywords('wc.18x2.g-fir.2x15-dexp.1')


@pytest.fixture
def simple_modelspec_with_phi(simple_modelspec):
    ms = from_keywords('wc.18x2.g-fir.2x15-dexp.1')
    return set_mean_phi(ms)


@pytest.fixture
def simple_recording():
    stim = np.random.rand(18, 200)
    resp = np.random.rand(1, 200)
    return Recording.load_from_arrays([stim, resp], 'simple_recording', 100,
                                      sig_names=['stim', 'resp'])
