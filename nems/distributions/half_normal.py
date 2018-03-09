from scipy import stats

from .distribution import Distribution


class HalfNormal(Distribution):
    '''
    HalfNormal prior

    Parameters
    ----------
    sigma : scalar or ndarray
        Standard deviation of distribution
    '''

    def __init__(self, sd, shape=None):
        self.sd = sd
        self.distribution = stats.halfnorm(scale=self.sd)

    def __repr__(self):
        sd = self.value_to_string(self.sd)
        return 'HalfNormal(σ={})'.format(sd)
