# For testing the predcitive accuracy of a set of modelspecs
import numpy as np
import copy

import nems.modelspec as ms
import nems.metrics.api as nmet
import nems.recording as recording


def generate_prediction(est, val, modelspecs):
    list_val = False
    if type(val) is list:
        # ie, if jackknifing
        list_val = True
    else:
        # Evaluate estimation and validation data

        # Since ms.evaluate only does a shallow copy of rec, successive
        # evaluations of one rec on many modelspecs just results in a list of
        # different pointers to the same recording. So need to force copies of
        # est/val before evaluating.
        if len(modelspecs) == 1:
            # no copies needed for 1 modelspec
            est = [est]
            val = [val]
        else:
            est = [est.copy() for i, _ in enumerate(modelspecs)]
            val = [val.copy() for i, _ in enumerate(modelspecs)]

    new_est = [ms.evaluate(d, m) for m, d in zip(modelspecs, est)]
    new_val = [ms.evaluate(d, m) for m, d in zip(modelspecs, val)]
    if list_val:
        new_val = [recording.jackknife_inverse_merge(new_val)]

    return new_est, new_val


def standard_correlation(est, val, modelspecs, rec=None):

    # Compute scores for validation dat
    r_ceiling = 0
    if len(val) == 1:
        r_test, se_test = nmet.j_corrcoef(val[0], 'pred', 'resp')
        r_fit, se_fit = nmet.j_corrcoef(est[0], 'pred', 'resp')
        r_floor = nmet.r_floor(val[0], 'pred', 'resp')
        if rec is not None:
            # print('running r_ceiling')
            r_ceiling = nmet.r_ceiling(val[0], rec, 'pred', 'resp')

    else:
        r = [nmet.corrcoef(p, 'pred', 'resp') for p in val]
        r_test = np.mean(r)
        se_test = np.std(r) / np.sqrt(len(val))
        r = [nmet.corrcoef(p, 'pred', 'resp') for p in est]
        r_fit = np.mean(r)
        se_fit = np.std(r) / np.sqrt(len(val))
        r_floor = [nmet.r_floor(p, 'pred', 'resp') for p in val]

        # TODO compute r_ceiling for multiple val sets
        r_ceiling = 0

    mse_test = [nmet.nmse(p, 'pred', 'resp') for p in val]
    ll_test = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in val]

    mse_fit = [nmet.nmse(p, 'pred', 'resp') for p in val]
    ll_fit = [nmet.likelihood_poisson(p, 'pred', 'resp') for p in est]

    modelspecs[0][0]['meta']['r_test'] = r_test
    modelspecs[0][0]['meta']['se_test'] = se_test
    modelspecs[0][0]['meta']['r_floor'] = r_floor
    modelspecs[0][0]['meta']['mse_test'] = np.mean(mse_test)
    modelspecs[0][0]['meta']['ll_test'] = np.mean(ll_test)

    modelspecs[0][0]['meta']['r_fit'] = r_fit
    modelspecs[0][0]['meta']['se_fit'] = se_fit
    modelspecs[0][0]['meta']['r_ceiling'] = r_ceiling
    modelspecs[0][0]['meta']['mse_fit'] = np.mean(mse_fit)
    modelspecs[0][0]['meta']['ll_fit'] = np.mean(ll_fit)

    return modelspecs


def correlation_per_model(est, val, modelspecs, rec=None):
    '''
    Expects the lengths of est, val, and modelspecs to match since est[i]
    should have been evaluated on the fitted modelspecs[i], etc.
    Similar to standard_correlation, but saves correlation information
    to every first-module 'meta' entry instead of saving an average
    to only the first modelspec
    '''
    if not len(est) == len(val) == len(modelspecs):
        raise ValueError("est, val, and modelspecs should all be lists"
                         " of equal length. got: %d, %d, %d respectively.",
                         len(est), len(val), len(modelspecs))

    modelspecs = copy.deepcopy(modelspecs)

    r_tests = [nmet.corrcoef(v, 'pred', 'resp') for v in val]
    #se_tests = [np.std(r)/np.sqrt(len(v)) for r, v in zip(r_tests, val)]
    mse_tests = [nmet.nmse(v, 'pred', 'resp') for v in val]
    ll_tests = [nmet.likelihood_poisson(v, 'pred', 'resp') for v in val]

    r_fits = [nmet.corrcoef(e, 'pred', 'resp') for e in est]
    #se_fits = [np.std(r)/np.sqrt(len(v)) for r, v in zip(r_fits, val)]
    mse_fits = [nmet.nmse(e, 'pred', 'resp') for e in est]
    ll_fits = [nmet.likelihood_poisson(e, 'pred', 'resp') for e in est]

    r_floors = [nmet.r_floor(v, 'pred', 'resp') for v in val]
    if rec is None:
        r_ceilings = [None]*len(r_floors)
    else:
        r_ceilings = [nmet.r_ceiling(v, rec, 'pred', 'resp') for v in val]

    for i, m in enumerate(modelspecs):
        m[0]['meta'].update({
                'r_test': r_tests[i], #'se_test': se_tests[i],
                'mse_test': mse_tests[i], 'll_test': ll_tests[i],
                'r_fit': r_fits[i], #'se_fit': se_fits[i],
                'mse_fit': mse_fits[i], 'll_fit': ll_fits[i],
                'r_floor': r_floors[i], 'r_ceiling': r_ceilings[i],
                })

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
