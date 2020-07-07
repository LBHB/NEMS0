import numpy as np


def sum_channels(rec, i, o):
    '''
    (NaN-)sums all the channels together in signal i and saves it to signal o.
    '''
    fn = lambda x: np.nansum(x, axis=0, keepdims=True)
    return [rec[i].transform(fn, o)]
