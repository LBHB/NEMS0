import copy
import logging
import time
from functools import partial
from nems.fitters.api import coordinate_descent, scipy_minimize
import nems.priors
import nems.fitters.mappers
import nems.modelspec as ms
import nems.metrics.api as metrics
import nems.segmentors
import nems.utils

import numpy as np

log = logging.getLogger(__name__)


def fit_basic(data, modelspec,
              fitter=scipy_minimize, cost_function=None,
              segmentor=nems.segmentors.use_all_data,
              mapper=nems.fitters.mappers.simple_vector,
              metric=lambda data: metrics.nmse(data, 'pred', 'resp'),
              metaname='fit_basic', fit_kwargs={}, require_phi=True):
    '''
    Required Arguments:
     data          A recording object
     modelspec     A modelspec object

    Optional Arguments:
     fitter        A function of (sigma, costfn) that tests various points,
                   in fitspace (i.e. sigmas) using the cost function costfn,
                   and hopefully returns a better sigma after some time.
     mapper        A class that has two methods, pack and unpack, which define
                   the mapping between modelspecs and a fitter's fitspace.
     segmentor     An function that selects a subset of the data during the
                   fitting process. This is NOT the same as est/val data splits
     metric        A function of a Recording that returns an error value
                   that is to be minimized.

    Returns
    A list containing a single modelspec, which has the best parameters found
    by this fitter.
    '''
    start_time = time.time()

    if cost_function is None:
        # Use the cost function defined in this module by default
        cost_function = basic_cost

    if require_phi:
        # Ensure that phi exists for all modules;
        # choose prior mean if not found
        for i, m in enumerate(modelspec):
            if ('phi' not in m.keys()) and ('prior' in m.keys()):
                log.debug('Phi not found for module, using mean of prior: %s',
                          m)
                m = nems.priors.set_mean_phi([m])[0]  # Inits phi for 1 module
                modelspec[i] = m

    if 'mask' in data.signals.keys():
        log.info("Data len pre-mask: %d", data['mask'].shape[1])
        data = data.apply_mask()
        log.info("Data len post-mask: %d", data['mask'].shape[1])

    ms.fit_mode_on(modelspec)

    # Create the mapper object that translates to and from modelspecs.
    # It has two methods that, when defined as mathematical functions, are:
    #    .pack(modelspec) -> fitspace_point
    #    .unpack(fitspace_point) -> modelspec
    packer, unpacker = mapper(modelspec)

    # A function to evaluate the modelspec on the data
    evaluator = nems.modelspec.evaluate

    my_cost_function = cost_function
    my_cost_function.counter = 0

    # Freeze everything but sigma, since that's all the fitter should be
    # updating.
    cost_fn = partial(my_cost_function,
                      unpacker=unpacker, modelspec=modelspec,
                      data=data, segmentor=segmentor, evaluator=evaluator,
                      metric=metric)

    # get initial sigma value representing some point in the fit space,
    # and corresponding bounds for each value
    sigma = packer(modelspec)
    bounds_to_vec, _ = nems.fitters.mappers.bounds_vector(modelspec)
    bounds = bounds_to_vec(modelspec)

    # Results should be a list of modelspecs
    # (might only be one in list, but still should be packaged as a list)
    improved_sigma = fitter(sigma, cost_fn, bounds=bounds, **fit_kwargs)
    improved_modelspec = unpacker(improved_sigma)

    elapsed_time = (time.time() - start_time)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.fit_mode_off(improved_modelspec)
    ms.set_modelspec_metadata(improved_modelspec, 'fitter', metaname)
    ms.set_modelspec_metadata(improved_modelspec, 'fit_time', elapsed_time)
    ms.set_modelspec_metadata(improved_modelspec, 'n_parms',
                              len(improved_sigma))

    results = [copy.deepcopy(improved_modelspec)]
    return results


def basic_cost(sigma, unpacker, modelspec, data, segmentor,
               evaluator, metric):
    '''Stanard cost function for use by fit_basic and other analyses.'''
    updated_spec = unpacker(sigma)
    # The segmentor takes a subset of the data for fitting each step
    # Intended use is for CV or random selection of chunks of the data
    # For fit_basic the 'segmentor' just passes it all through.
    data_subset = segmentor(data)
    updated_data_subset = evaluator(data_subset, updated_spec)
    error = metric(updated_data_subset)
    log.debug("inside cost function, current error: %.06f", error)
    log.debug("current sigma: %s", sigma)

    if hasattr(basic_cost, 'counter'):
        basic_cost.counter += 1
        if basic_cost.counter % 500 == 0:
            log.info('Eval #%d. E=%.06f', basic_cost.counter, error)
            # log.debug("current sigma: %s", sigma)
            nems.utils.progress_fun()

    if hasattr(basic_cost, 'error'):
        basic_cost.error = error

    return error


def fit_random_subsets(data, modelspec, nsplits=1, rebuild_every=10000):
    '''
    Randomly picks a small fraction of the data to fit on.
    Intended to speed up initial converge on fitting large data sets.
    To improve efficiency, you may generally good to use the same subset
    for a bunch of cost function evaluations in a row.
    '''
    maker = nems.segmentors.random_jackknife_maker
    segmentor = maker(nsplits=nsplits, rebuild_every=rebuild_every,
                      invert=True, excise=True)
    return fit_basic(data, modelspec,
                     segmentor=segmentor,
                     metaname='fit_random_subsets')


def fit_nfold(data_list, modelspecs, generate_psth=False,
              fitter=scipy_minimize,
              metric=None,
              fit_kwargs={'options': {'ftol': 1e-7, 'maxiter': 1000}}):
    '''
    Takes njacks jackknifes, where each jackknife has some small
    fraction of data NaN'd out, and fits modelspec to them.

    TESTING:
    if input len(modelspecs) == len(data_list) then use each
      modelspec as initial condition for corresponding data_list fold
    if len(modelspecs) == 1, then use the same initial conditions for
      each fold

    '''
    nfolds = len(data_list)
    models = []
    if not metric:
        metric = lambda d: metrics.nmse(d, 'pred', 'resp')

    for i in range(nfolds):
        if len(modelspecs) > 1:
            msidx = i
        else:
            msidx = 0

        log.info("Fitting fold %d/%d from modelspec %d", i+1, nfolds, msidx)
#        resp = data_list[i]['resp']
#        resp_len = np.sum(np.isfinite(resp.as_continuous()))
#        log.info("non-nan resp samples: %d", resp_len)
#        stim = data_list[i]['psth']
#        stim_len = np.sum(np.isfinite(stim.as_continuous()[0, :]))
#        log.info("non-nan stim samples: %d", stim_len)
        models += fit_basic(data_list[i], copy.deepcopy(modelspecs[msidx]),
                            fitter=fitter,
                            metric=metric,
                            metaname='fit_nfold',
                            fit_kwargs=fit_kwargs)

    return models


def fit_state_nfold(data_list, modelspecs, generate_psth=False,
              fitter=scipy_minimize,
              metric=None,
              fit_kwargs={'options': {'ftol': 1e-7, 'maxiter': 1000}}):
    '''
    Generic state-dependent-stream model fitter
    Takes njacks jackknifes, where each jackknife has some small
    fraction of data NaN'd out, and fits modelspec to them.

    DEPRECATED? REPLACED BY STANDARD nfold?
    '''
    nfolds = len(data_list)

    models = []
    if not metric:
        metric = lambda d: metrics.nmse(d, 'pred', 'resp')

    for i in range(nfolds):
        log.info("Fitting fold {}/{}".format(i+1, nfolds))
        tms = nems.initializers.prefit_to_target(
                data_list[i], copy.deepcopy(modelspecs[0]),
                nems.analysis.api.fit_basic, 'merge_channels',
                fitter=scipy_minimize,
                fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})

        models += fit_basic(data_list[i], tms,
                            fitter=fitter,
                            metric=metric,
                            metaname='fit_nfold',
                            fit_kwargs=fit_kwargs)

    return models


def fit_jackknifes(data, modelspec, njacks=10):
    '''
    Takes njacks jackknifes, where each jackknife has some small
    fraction of data NaN'd out, and fits modelspec to them.

    TODO : check if deprecated, replaced by fit_nfold?
    '''
    models = []
    for i in range(njacks):
        log.info("Fitting jackknife {}/{}".format(i+1, njacks))
        jk = data.jackknife_by_time(njacks, i)
        models += fit_basic(jk, modelspec, fitter=scipy_minimize,
                            metaname='fit_jackknifes')

    return models


def fit_subsets(data, modelspec, nsplits=10):
    '''
    Divides the data evenly into nsplits pieces, and fits a model
    to each of the pieces.

    TODO : Test, add more parameters
    '''
    models = []
    for i in range(nsplits):
        # TODO: Minor glitch: when fitting, print output from fitter
        #       comes back *after* log from next iteration
        #       (i.e. "fitting 1/3"
        #             "fitting 2/3"
        #             "final error <for 1/3>: 0.111")
        log.info("Fitting subset {}/{}".format(i+1, nsplits))
        split = data.jackknife_by_time(nsplits, i, invert=True, excise=True)
        models += fit_basic(split, modelspec, fitter=scipy_minimize,
                            metaname='fit_subset')

    return models


def fit_from_priors(data, modelspec, ntimes=10):
    '''
    Fit ntimes times, starting from random points sampled from the prior.

    TODO : Test, add more parameters
    '''
    models = []
    for i in range(ntimes):
        log.info("Fitting from random start: {}/{}".format(i+1, ntimes))
        ms = nems.priors.set_random_phi(modelspec)
        models += fit_basic(data, ms, fitter=scipy_minimize,
                            metaname='fit_from_priors')

    return models
