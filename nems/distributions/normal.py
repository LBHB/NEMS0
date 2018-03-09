from scipy import stats

from .distribution import Distribution


class Normal(Distribution):
    '''
    Normal prior

    Parameters
    ----------
    mu : scalar or ndarray
        Mean of distribution
    sd : scalar or ndarray
        Standard deviation of distribution

    Example
    -------
    Define a scalar prior for a single coefficient
    >>> weights = Normal(mu=3, sd=5)
    >>> weights.mean()
    3

    Define an array of priors with different means but same standard
    deviation
    >>> weights = Normal(mu=[1, 5], sd=[1, 1])
    >>> weights.mean()
    [1, 5]

    Define an array of priors with same mean and standard deviation
    >>> weights = Normal(mu=[3, 3], sd=[1, 1])
    >>> weights.mean()
    [3, 3]
    '''

    def __init__(self, mu, sd):
        self.mu = mu
        self.sd = sd

        # The beauty of using the scipy stats module is it correctly handles
        # scalar and multidimensional arrays of distributions.
        self.distribution = stats.norm(loc=self.mu, scale=self.sd)

    def __repr__(self):
        return str(self.tolist())
