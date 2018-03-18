# For testing the predcitive accuracy of a set of modelspecs
import numpy as np

import nems.modelspec as ms
import nems.metrics.api as nmet


def standard_correlation(est, val, modelspecs):
    # Evaluate estimation data, then store the results
    # of metric tests in first modelspec's meta field.
    new_rec = [ms.evaluate(val, m) for m in modelspecs]
    r_test = [nmet.corrcoef(p, 'pred', 'resp') for p in new_rec]
    mse_test = [nmet.nmse(p, 'pred', 'resp') for p in new_rec]
    ll_test = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in new_rec]
    modelspecs[0][0]['meta']['r_test'] = np.mean(r_test)
    modelspecs[0][0]['meta']['mse_test'] = np.mean(mse_test)
    modelspecs[0][0]['meta']['ll_test'] = np.mean(ll_test)

    # Repeat for validation data.
    new_rec = [ms.evaluate(est, m) for m in modelspecs]
    r_fit = [nmet.corrcoef(p, 'pred', 'resp') for p in new_rec]
    mse_fit = [nmet.nmse(p, 'pred', 'resp') for p in new_rec]
    ll_fit = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in new_rec]
    modelspecs[0][0]['meta']['r_fit'] = np.mean(r_fit)
    modelspecs[0][0]['meta']['mse_fit'] = np.mean(mse_fit)
    modelspecs[0][0]['meta']['ll_fit'] = np.mean(ll_fit)

    return modelspecs
