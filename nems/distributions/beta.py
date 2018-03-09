import numpy as np
from scipy import stats

from .distribution import Distribution


class Beta(Distribution):
    '''
    Beta prior

    Parameters
    ----------
    alpha : scalar or ndarray
        Parameter of distribution
    beta : scalar or ndarray
        Parameter of distribution
    '''

    def __init__(self, alpha, beta):
        self.alpha = np.asarray(alpha)
        self.beta = np.asarray(beta)

        # The beauty of using the scipy stats module is it correctly handles
        # scalar and multidimensional arrays of distributions.
        self.distribution = stats.beta(self.alpha, self.beta)

    def __repr__(self):
        alpha = self.value_to_string(self.alpha)
        beta = self.value_to_string(self.beta)
        return 'Beta(α={}, β={})'.format(alpha, beta)
