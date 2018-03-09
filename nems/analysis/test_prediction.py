# For testing the predcitive accuracy of a set of modelspecs

# TODO: Rework this code snippet to use actual metrics

# Take your final measurements on the validation data set, and save!
# results = {}
# for m in nems.metrics.get_some_list_of_metric_functions:
#    results[m.name] = m(val['resp'], evaluator(val, modelspec_fitted))

import nems.modelspec as ms
import nems.metrics.api
import numpy as np

def standard_correlation(est,val,modelspecs):
    
    new_rec = [ms.evaluate(val, m) for m in modelspecs]
    r_test = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_rec]
    new_rec = [ms.evaluate(est, m) for m in modelspecs]
    r_fit = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_rec]
    modelspecs[0][0]['meta']['r_fit']=np.mean(r_fit)
    modelspecs[0][0]['meta']['r_test']=np.mean(r_test)
    
    return modelspecs
