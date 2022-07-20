from scipy import stats
import numpy as np

from .distribution import Distribution


class Uniform(Distribution):
    '''
    Uniform prior

    Parameters
    ----------
    lower : scalar or ndarray
        Lower bound of distribution
    upper : scalar or ndarray
        Upper bound of distribution
    '''

    def __init__(self, lower, upper):
        self._lower = np.asarray(lower)
        self._upper = np.asarray(upper)

        # The beauty of using the scipy stats module is it correctly handles
        # scalar and multidimensional arrays of distributions.
        self.distribution = stats.uniform(loc=self._lower,
                                          scale=self._upper-self._lower)

    def __repr__(self):
        lower = self.value_to_string(self._lower)
        upper = self.value_to_string(self._upper)
        return 'Uniform({}, {})'.format(lower, upper)
