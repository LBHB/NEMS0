import numpy as np
from numpy.random import binomial, uniform

def joint_prob_indep(data, n=1000):
    """
    test whether a joint binomial probability distribution is significantly different from the produce of marginals
    permutation test. ie, p(x1=True) x p(x2=True) == p(x1=True, x2=True)
    :param data: 2x2 matrix of x1, x2 outcomes. p(x2=True)
    :param n: number of permutations
    :return: m: mean predicted p(True,True) from marginal, p: probability that actual data occurred from marginals
    """

    T = np.sum(data)
    p1 = np.sum(data[1,:])/T
    p2 = np.sum(data[:,1])/T

    _d = np.zeros((2,2,n))
    for i in range(n):
        _d1 = (uniform(0,1,T)<p1).astype(int)
        _d2 = (uniform(0,1,T)<p2).astype(int)
        _d[0,0,i] = np.sum((_d1==0) & (_d2==0))
        _d[1,0,i] = np.sum((_d1==1) & (_d2==0))
        _d[0,1,i] = np.sum((_d1==0) & (_d2==1))
        _d[1,1,i] = np.sum((_d1==1) & (_d2==1))

    nj = _d[1,1,:]

    m = nj.mean()
    p = np.sum(nj<data[1,1])/n

    return m, p