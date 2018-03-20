# For testing the predcitive accuracy of a set of modelspecs
import numpy as np

import nems.modelspec as ms
import nems.metrics.api as nmet

def standard_correlation(est,val,modelspecs):

    if type(est) is list:
        # ie, if jackknifing
        new_val = [ms.evaluate(d, m) for m,d in zip(modelspecs,val)]
        new_est = [ms.evaluate(d, m) for m,d in zip(modelspecs,est)]
        new_val=[new_val[0].jackknife_inverse_merge(new_val)]
        #r_test = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_val]

        r_test = [nmet.corrcoef(new_val[0], 'pred', 'resp')]
        mse_test = [nmet.nmse(new_val[0], 'pred', 'resp')]
        ll_test = [nmet.likelihood_poisson(new_val[0], 'pred', 'resp')]

        r_fit = [nmet.corrcoef(p, 'pred', 'resp') for p in new_est]
        mse_fit = [nmet.nmse(p, 'pred', 'resp') for p in new_est]
        ll_fit = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in new_est]

    else:
        # Evaluate estimation data, then store the results
        # of metric tests in first modelspec's meta field.
        new_val = [ms.evaluate(val, m) for m in modelspecs]
        r_test = [nmet.corrcoef(p, 'pred', 'resp') for p in new_val]
        mse_test = [nmet.nmse(p, 'pred', 'resp') for p in new_val]
        ll_test = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in new_val]

        # Repeat for validation data.
        new_est = [ms.evaluate(est, m) for m in modelspecs]
        r_fit = [nmet.corrcoef(p, 'pred', 'resp') for p in new_est]
        mse_fit = [nmet.nmse(p, 'pred', 'resp') for p in new_est]
        ll_fit = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in new_est]

    modelspecs[0][0]['meta']['r_test'] = np.mean(r_test)
    modelspecs[0][0]['meta']['mse_test'] = np.mean(mse_test)
    modelspecs[0][0]['meta']['ll_test'] = np.mean(ll_test)

    modelspecs[0][0]['meta']['r_fit'] = np.mean(r_fit)
    modelspecs[0][0]['meta']['mse_fit'] = np.mean(mse_fit)
    modelspecs[0][0]['meta']['ll_fit'] = np.mean(ll_fit)

    return modelspecs,new_est,new_val

