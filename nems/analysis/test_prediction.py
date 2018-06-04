# For testing the predcitive accuracy of a set of modelspecs
import numpy as np
import copy

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

    return new_est, new_val


def standard_correlation(est, val, modelspecs, rec=None):

    # Compute scores for validation data
    r_test = [nmet.corrcoef(p, 'pred', 'resp') for p in val]
    mse_test = [nmet.nmse(p, 'pred', 'resp') for p in val]
    ll_test = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in val]

    # Repeat for est data.
    r_fit = [nmet.corrcoef(p, 'pred', 'resp') for p in est]
    r_floor = [nmet.r_floor(p, 'pred', 'resp') for p in val]
    if rec is not None:
        #print('running r_ceiling')
        r_ceiling = [nmet.r_ceiling(p, rec, 'pred', 'resp') for p in val]
    else:
        #log.info('skipping r_ceiling')
        pass
    mse_fit = [nmet.nmse(p, 'pred', 'resp') for p in val]
    ll_fit = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in est]
    print(r_ceiling,r_test)
    modelspecs[0][0]['meta']['r_test'] = np.mean(r_test)
    modelspecs[0][0]['meta']['mse_test'] = np.mean(mse_test)
    modelspecs[0][0]['meta']['ll_test'] = np.mean(ll_test)

    modelspecs[0][0]['meta']['r_fit'] = np.mean(r_fit)
    modelspecs[0][0]['meta']['r_floor'] = np.mean(r_floor)
    if rec is not None:
        modelspecs[0][0]['meta']['r_ceiling'] = np.mean(r_ceiling)
    modelspecs[0][0]['meta']['mse_fit'] = np.mean(mse_fit)
    modelspecs[0][0]['meta']['ll_fit'] = np.mean(ll_fit)

    return modelspecs

def standard_correlation_by_epochs(est,val,modelspecs,epochs_list, rec=None):
    
    #Does the same thing as standard_correlation, excpet with subsets of data
    #defined by epochs_list
    
    #To use this, first add epochs to define subsets of data.
    #Then, pass epochs_list as a list of subsets to test.
    #For example, ['A', 'B', ['A', 'B']] will measure correlations separately
    # for all epochs marked 'A', all epochs marked 'B', and all epochs marked 
    # 'A'or 'B'
    
    
    for epochs in epochs_list:
        # Create a label for this subset. If epochs is a list, join elements with "+"
        epoch_list_str="+".join([str(x) for x in epochs])

        # Make a copy for this subset
        val_copy=copy.deepcopy(val)
        for vc in val_copy:
            vc['resp']=vc['resp'].select_epochs(epochs)

        est_copy=copy.deepcopy(est)
        for ec in est_copy:
            ec['resp']=ec['resp'].select_epochs(epochs)

        # Compute scores for validation data
        r_test = [nmet.corrcoef(p, 'pred', 'resp') for p in val_copy]
        mse_test = [nmet.nmse(p, 'pred', 'resp') for p in val_copy]
        ll_test = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in val_copy]

        r_floor = [nmet.r_floor(p, 'pred', 'resp') for p in val]
        if rec is not None:
            r_ceiling = [nmet.r_ceiling(p, rec, 'pred', 'resp') for p in val_copy]

        # Repeat for est data.
        r_fit = [nmet.corrcoef(p, 'pred', 'resp') for p in est_copy]
        mse_fit = [nmet.nmse(p, 'pred', 'resp') for p in est_copy]
        ll_fit = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in est_copy]

        #Avergage
        modelspecs[0][0]['meta'][epoch_list_str]={}
        modelspecs[0][0]['meta'][epoch_list_str]['r_test'] = np.mean(r_test)
        modelspecs[0][0]['meta'][epoch_list_str]['mse_test'] = np.mean(mse_test)
        modelspecs[0][0]['meta'][epoch_list_str]['ll_test'] = np.mean(ll_test)

        modelspecs[0][0]['meta'][epoch_list_str]['r_fit'] = np.mean(r_fit)
        modelspecs[0][0]['meta'][epoch_list_str]['r_floor'] = np.mean(r_floor)
        if rec is not None:
            modelspecs[0][0]['meta'][epoch_list_str]['r_ceiling'] = np.mean(r_ceiling)
        modelspecs[0][0]['meta'][epoch_list_str]['mse_fit'] = np.mean(mse_fit)
        modelspecs[0][0]['meta'][epoch_list_str]['ll_fit'] = np.mean(ll_fit)

    return modelspecs

def generate_prediction_sets(est, val, modelspecs):
    if type(val) is list:
        # ie, if jackknifing
        new_est = [ms.evaluate(d, m) for m,d in zip(modelspecs,est)]
        new_val = [ms.evaluate(d, m) for m,d in zip(modelspecs,val)]
    else:
        print('val and est must be lists')

    return new_est, new_val


def standard_correlation_by_set(est, val, modelspecs):

    # Compute scores for validation data
    r_test = [nmet.corrcoef(p, 'pred', 'resp') for p in val]
    mse_test = [nmet.nmse(p, 'pred', 'resp') for p in val]
    ll_test = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in val]

    # Repeat for est data.
    r_fit = [nmet.corrcoef(p, 'pred', 'resp') for p in est]
    mse_fit = [nmet.nmse(p, 'pred', 'resp') for p in est]
    ll_fit = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in est]
    for i in range(len(modelspecs)):
        modelspecs[i][0]['meta']['r_test'] = r_test[i]
        modelspecs[i][0]['meta']['mse_test'] = mse_test[i]
        modelspecs[i][0]['meta']['ll_test'] = ll_test[i]

        modelspecs[i][0]['meta']['r_fit'] = r_fit[i]
        modelspecs[i][0]['meta']['mse_fit'] = mse_fit[i]
        modelspecs[i][0]['meta']['ll_fit'] = ll_fit[i]

    return modelspecs
