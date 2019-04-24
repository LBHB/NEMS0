# For testing the predcitive accuracy of a set of modelspecs
import numpy as np
import copy
import logging

import nems.modelspec as ms
import nems.metrics.api as nmet
from nems.analysis.cost_functions import basic_cost
import nems.priors
import nems.fitters.mappers
import nems.segmentors
import nems.utils

log = logging.getLogger(__name__)


def generate_prediction(est, val, modelspec, **context):

    list_val = (type(val) is list)
    list_modelspec = (type(modelspec) is list)
    if list_modelspec:
        modelspecs = modelspec
    else:
        modelspecs = modelspec.fits()

    if list_val:
        raise ValueError("list type val recordings no longer supported")

    # Evaluate estimation and validation data

    # three scenarios:
    # 1 fit, 1 est set - standard
    # n fits, n est sets - nfold
    # n fits, 1 est set - multiple initial conditions

    if est.view_count() == 1:
        new_est = est.tile_views(len(modelspecs))
        new_val = val.tile_views(len(modelspecs))
        do_inverse_merge = False
    else:
        # assume est and val have view_count() == len(modelspecs)
        new_est = est.copy()
        new_val = val.copy()
        do_inverse_merge = True

    for i, m in enumerate(modelspecs):
        # update each view with prediction from corresponding modelspec
        new_est = ms.evaluate(new_est.set_view(i), m)
        new_val = ms.evaluate(new_val.set_view(i), m)

        # this seems kludgy. but where should mask be handled?
        if 'mask' in new_val.signals.keys():
            m = new_val['mask'].as_continuous()
            x = new_val['pred'].as_continuous().copy()
            x[..., m[0,:] == 0] = np.nan
            new_val['pred'] = new_val['pred']._modified_copy(x)

    if do_inverse_merge:
        new_val = new_val.jackknife_inverse_merge()

    return new_est, new_val


def standard_correlation(est, val, modelspec=None, modelspecs=None, rec=None,
                         output_name='resp', use_mask=True):
    # use_mask: mask before computing metrics (if mask exists)
    # Compute scores for validation dat
    r_ceiling = 0

    # some crazy stuff to maintain backward compatibility
    # eventually we will only support modelspec and deprecate support for
    # modelspecs lists
    if modelspecs is None:
        list_modelspec = (type(modelspec) is list)
        if modelspec is None:
            raise ValueError('modelspecs or modelspec required for input')
        if list_modelspec:
            modelspecs = modelspec
        else:
            modelspecs = modelspec.fits()
    else:
        list_modelspec = True

    if type(val) is not list:

        # TODO: support for views

        if ('mask' in val.signals.keys()) and use_mask:
            v = val.apply_mask()
            e = est.apply_mask()
        else:
            v = val
            e = est
            use_mask = False

        r_test, se_test = nmet.j_corrcoef(v, 'pred', output_name)
        r_fit, se_fit = nmet.j_corrcoef(e, 'pred', output_name)
        r_floor = nmet.r_floor(v, 'pred', output_name)
        if rec is not None:
            if 'mask' in rec.signals.keys() and use_mask:
                r = rec.apply_mask()
            else:
                r = rec
            # print('running r_ceiling')
            r_ceiling = nmet.r_ceiling(v, r, 'pred', output_name)

        mse_test, se_mse_test = nmet.j_nmse(v, 'pred', output_name)
        mse_fit, se_mse_fit = nmet.j_nmse(e, 'pred', output_name)

        ll_test = nmet.likelihood_poisson(v, 'pred', output_name)
        ll_fit = nmet.likelihood_poisson(e, 'pred', output_name)

    elif len(val) == 1:
        # does this ever run?
        raise ValueError("val as list not supported any more?")
        if ('mask' in val[0].signals.keys()) and use_mask:
            v = val[0].apply_mask()
            e = est[0].apply_mask()
        else:
            v = val[0]
            e = est[0]

        r_test, se_test = nmet.j_corrcoef(v, 'pred', output_name)
        r_fit, se_fit = nmet.j_corrcoef(e, 'pred', output_name)
        r_floor = nmet.r_floor(v, 'pred', output_name)
        if rec is not None:
            try:
                # print('running r_ceiling')
                r_ceiling = nmet.r_ceiling(v, rec, 'pred', output_name)
            except:
                r_ceiling = 0

        mse_test, se_mse_test = nmet.j_nmse(v, 'pred', output_name)
        mse_fit, se_mse_fit = nmet.j_nmse(e, 'pred', output_name)

        ll_test = nmet.likelihood_poisson(v, 'pred', output_name)
        ll_fit = nmet.likelihood_poisson(e, 'pred', output_name)

    else:
        # unclear if this ever excutes since jackknifed val sets are
        # typically already merged
        raise ValueError("no support for val list of recordings len>1")
        r = [nmet.corrcoef(p, 'pred', output_name) for p in val]
        r_test = np.mean(r)
        se_test = np.std(r) / np.sqrt(len(val))
        r = [nmet.corrcoef(p, 'pred', output_name) for p in est]
        r_fit = np.mean(r)
        se_fit = np.std(r) / np.sqrt(len(val))
        r_floor = [nmet.r_floor(p, 'pred', output_name) for p in val]

        # TODO compute r_ceiling for multiple val sets
        r_ceiling = 0

        mse_test = [nmet.nmse(p, 'pred', output_name) for p in val]
        mse_fit = [nmet.nmse(p, 'pred', output_name) for p in est]

        se_mse_test = np.std(mse_test) / np.sqrt(len(val))
        se_mse_fit = np.std(mse_fit) / np.sqrt(len(est))
        mse_test = np.mean(mse_test)
        mse_fit = np.mean(mse_fit)

        ll_test = np.mean([nmet.likelihood_poisson(p, 'pred', output_name) for p in val])
        ll_fit = np.mean([nmet.likelihood_poisson(p, 'pred', output_name) for p in est])

    modelspecs[0][0]['meta']['r_test'] = r_test
    modelspecs[0][0]['meta']['se_test'] = se_test
    modelspecs[0][0]['meta']['r_floor'] = r_floor
    modelspecs[0][0]['meta']['mse_test'] = mse_test
    modelspecs[0][0]['meta']['se_mse_test'] = se_mse_test
    modelspecs[0][0]['meta']['ll_test'] = ll_test

    modelspecs[0][0]['meta']['r_fit'] = r_fit
    modelspecs[0][0]['meta']['se_fit'] = se_fit
    modelspecs[0][0]['meta']['r_ceiling'] = r_ceiling
    modelspecs[0][0]['meta']['mse_fit'] = mse_fit
    modelspecs[0][0]['meta']['se_mse_fit'] = se_mse_fit
    modelspecs[0][0]['meta']['ll_fit'] = ll_fit

    if list_modelspec:
        # backward compatibility
        return modelspecs
    else:
        return modelspecs[0]


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


def standard_correlation_by_epochs(est,val,modelspec=None,modelspecs=None,epochs_list=None, rec=None):

    #Does the same thing as standard_correlation, excpet with subsets of data
    #defined by epochs_list

    #To use this, first add epochs to define subsets of data.
    #Then, pass epochs_list as a list of subsets to test.
    #For example, ['A', 'B', ['A', 'B']] will measure correlations separately
    # for all epochs marked 'A', all epochs marked 'B', and all epochs marked
    # 'A'or 'B'

    if modelspecs is None:
        list_modelspec = (type(modelspec) is list)
        if modelspec is None:
            raise ValueError('modelspecs or modelspec required for input')
        if list_modelspec:
            modelspecs = modelspec
        else:
            modelspecs = modelspec.fits()
    else:
        list_modelspec = True

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

    if list_modelspec:
        # backward compatibility
        return modelspecs
    else:
        return modelspecs[0]


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


def basic_error(data, modelspec, cost_function=None,
                segmentor=nems.segmentors.use_all_data,
                mapper=nems.fitters.mappers.simple_vector,
                metric=lambda data: nmet.nmse(data, 'pred', 'resp')):
    '''
    Similar to fit_basic except that it just returns the error for the fitting
    process instead of a modelspec. Intended to be called after a model
    has already been fit.
    '''
    modelspec = copy.deepcopy(modelspec)
    if cost_function is None:
        # Use the cost function defined in this module by default
        cost_function = basic_cost

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    if 'mask' in data.signals.keys():
        log.info("Data len pre-mask: %d", data['mask'].shape[1])
        data = data.apply_mask()
        log.info("Data len post-mask: %d", data['mask'].shape[1])

    packer, unpacker, pack_bounds = mapper(modelspec)
    evaluator = nems.modelspec.evaluate
    sigma = packer(modelspec)
    error = cost_function(sigma, unpacker, modelspec, data, segmentor,
                          evaluator, metric)

    return error
