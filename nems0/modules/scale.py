def scale(rec, i, o, a):
    '''
    Intended to be applied immediately preceding levelshift, so that the
    total output of the spectrotemporal filter is:
        y = a*x + levelshift

    Parameters
    ----------
    a : a scalar to multiply every element of the input signal by.
    '''
    fn = lambda x: x * a
    return [rec[i].transform(fn, o)]

def null(rec, i, o, **kwargs):
    """
    do nothing - can be subbed in for excluded module
    :param rec:
    :param i:
    :param o:
    :param kwargs:
    :return:
    """
    return []
