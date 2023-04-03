import numpy as np
from nems0.distributions.api import Normal, HalfNormal


# TODO: How do we integrate plotting tests?


def test_distributions():
    d1 = Normal(mean=[-0.9, 0.2], sd=[1, 0.4])
    d2 = HalfNormal(sd=[[1, 0.4, .5], [0.3, 0.1, .7]])

    n = 100
    d1_sample = d1.sample(n)
    d2_sample = d2.sample(n)
    assert(d1_sample.shape == (n, 2))
    assert(d2_sample.shape == (n, 2, 3))
