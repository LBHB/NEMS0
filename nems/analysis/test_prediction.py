# For testing the predcitive accuracy of a set of modelspecs
import numpy as np

import nems.modelspec as ms
import nems.metrics.api as nmet
import nems.recording as recording

def generate_prediction(est,val,modelspecs):

    if type(val) is list:
        # ie, if jackknifing
        new_est = [ms.evaluate(d, m) for m,d in zip(modelspecs,est)]
        new_val = [ms.evaluate(d, m) for m,d in zip(modelspecs,val)]
        new_val = [recording.jackknife_inverse_merge(new_val)]
    else:
        # Evaluate estimation and validation data
        new_est = [ms.evaluate(est, m) for m in modelspecs]
        new_val = [ms.evaluate(val, m) for m in modelspecs]

    return new_est,new_val


def standard_correlation(est, val, modelspecs, rec=None):

    # Compute scores for validation data
    r_test = [nmet.corrcoef(p, 'pred', 'resp') for p in val]
    mse_test = [nmet.nmse(p, 'pred', 'resp') for p in val]
    ll_test = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in val]

    # Repeat for est data.
    r_fit = [nmet.corrcoef(p, 'pred', 'resp') for p in est]
    r_floor = [nmet.r_floor(p, 'pred', 'resp') for p in val]
    if rec:
        r_ceiling = [nmet.r_ceiling(p, rec, 'pred', 'resp') for p in val]        
    mse_fit = [nmet.nmse(p, 'pred', 'resp') for p in val]
    ll_fit = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in est]

    modelspecs[0][0]['meta']['r_test'] = np.mean(r_test)
    modelspecs[0][0]['meta']['mse_test'] = np.mean(mse_test)
    modelspecs[0][0]['meta']['ll_test'] = np.mean(ll_test)

    modelspecs[0][0]['meta']['r_fit'] = np.mean(r_fit)
    modelspecs[0][0]['meta']['r_floor'] = np.mean(r_floor)
    modelspecs[0][0]['meta']['r_ceiling'] = np.mean(r_ceiling)
    modelspecs[0][0]['meta']['mse_fit'] = np.mean(mse_fit)
    modelspecs[0][0]['meta']['ll_fit'] = np.mean(ll_fit)

    return modelspecs

