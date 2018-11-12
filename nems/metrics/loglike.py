import numpy as np


def likelihood_poisson(result, pred_name='pred', resp_name='resp'):
    # TODO: Copied from old NEMS master, needs documentation.

    x1 = result[pred_name].as_continuous()
    x2 = result[resp_name].as_continuous()

    # only keep indices where neither array is NaN
    keepidx = np.isfinite(x1) * np.isfinite(x2)
    x1 = x1[keepidx]
    x2 = x2[keepidx]

    # TODO: Why set this limit?
    x1[x1 < 0.00001] = 0.00001

    # norm LL copied from NARF:
    # - nanmean(r.*log(p) - p) ./ (nanmean(r)*log(nanmean(r)));

    numer = np.mean(x2*np.log(x1) - x1)
    denom = np.mean(x2) * np.log(np.mean(x2))
    return numer/denom
