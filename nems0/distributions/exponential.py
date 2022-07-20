from scipy import stats
import numpy as np

from .distribution import Distribution


class Exponential(Distribution):
    '''
    Exponential Prior

    Parameters
    ----------
    beta : scalar or ndarray
        Scale of distribution.
        Also determines mean=beta and std=beta
    '''

    def __init__(self, beta):
        self._beta = self._mean = self._sd = np.asarray(beta)
        self.distribution = stats.expon(scale=beta)

    def __repr__(self):
        return 'Exponential(β={})'.format(self._beta)
