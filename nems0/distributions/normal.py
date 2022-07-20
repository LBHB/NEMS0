from scipy import stats
import numpy as np

from .distribution import Distribution


class Normal(Distribution):
    '''
    Normal prior

    Parameters
    ----------
    mean : scalar or ndarray
        Mean of distribution
    sd : scalar or ndarray
        Standard deviation of distribution

    Example
    -------
    Define a scalar prior for a single coefficient
    >>> weights = Normal(mean=3, sd=5)
    >>> weights.mean()
    3

    Define an array of priors with different means but same standard
    deviation
    >>> weights = Normal(mean=[1, 5], sd=[1, 1])
    >>> weights.mean()
    [1, 5]

    Define an array of priors with same mean and standard deviation
    >>> weights = Normal(mean=[3, 3], sd=[1, 1])
    >>> weights.mean()
    [3, 3]
    '''

    def __init__(self, mean, sd):
        self._mean = np.asarray(mean)
        self._sd = np.asarray(sd)

        # The beauty of using the scipy stats module is it correctly handles
        # scalar and multidimensional arrays of distributions.
        self.distribution = stats.norm(loc=self._mean, scale=self._sd)

    def __repr__(self):
        mean = self.value_to_string(self._mean)
        sd = self.value_to_string(self._sd)
        return 'Normal(μ={}, σ={})'.format(mean, sd)
