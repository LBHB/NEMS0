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
    
    if type(est) is list:
        # ie, if jackknifing
        new_val = [ms.evaluate(d, m) for m,d in zip(modelspecs,val)]
        new_est = [ms.evaluate(d, m) for m,d in zip(modelspecs,est)]
        new_val=new_val[0].jackknife_inverse_merge(new_val)
        #r_test = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_val]
        r_fit = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_est]
    else:
        new_val = [ms.evaluate(val, m) for m in modelspecs]
        new_est = [ms.evaluate(est, m) for m in modelspecs]
        r_test = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_val]
        r_fit = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_est]
        
    modelspecs[0][0]['meta']['r_fit']=np.mean(r_fit)
    modelspecs[0][0]['meta']['r_test']=np.mean(r_test)
    
    return modelspecs

