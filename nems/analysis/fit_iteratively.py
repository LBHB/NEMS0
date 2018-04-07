from functools import partial
import time
import logging
import copy

from nems.fitters.api import coordinate_descent, scipy_minimize
import nems.fitters.mappers
import nems.metrics.api
import nems.modelspec as ms

log = logging.getLogger(__name__)

'''
need extra parameters
    max_step_per_iter - max number of steps fitting each modules_set before moving to the next one

# code taken from nems_retired:

class fit_iteratively(nems_fitter):
    """
    iterate through modules, running fitting each one with sub_fitter()

    TODO: update class name to FitIteratively per pep8 guidelines.

    """

    name = 'fit_iteratively'
    sub_fitter = None
    max_iter = 100
    module_sets=[]
    tolerance=0.0000001  # deprecated
    start_tolerance=0.000001  # start tolerance
    min_tolerance=0.00000001  # end tolerance

    def my_init(self, sub_fitter=basic_min, max_iter=100,
                min_kwargs={'routine': 'L-BFGS-B', 'maxit': 10000},
                start_tolerance=0.00001, min_tolerance=0.00000001):
        self.sub_fitter = sub_fitter(self.stack, **min_kwargs)
        self.max_iter = max_iter
        self.module_sets = [[i] for i in self.fit_modules]

    def do_fit(self):
        self.sub_fitter.tolerance = self.start_tolerance
        itr = 0
        err = self.stack.error()
        this_itr = 0
        d_err=1
        while itr < self.max_iter and d_err>self.min_tolerance:
            this_itr += 1

            for i in self.module_sets:
                log.info("Begin sub_fitter on mod: {0}; iter {1}; tol={2}"
                         .format(self.stack.modules[i[0]].name, itr,
                                 self.sub_fitter.tolerance))
                self.sub_fitter.fit_modules = i
                new_err = self.sub_fitter.do_fit()

            d_err=err-new_err

            if d_err < self.sub_fitter.tolerance and d_err>self.min_tolerance:
                log.info("\nIter {0}: error improvement < tol {1}, "
                         "starting new outer iteration"
                         .format(itr,self.sub_fitter.tolerance))
                itr += 1
                self.sub_fitter.tolerance = self.sub_fitter.tolerance / 2
                this_itr = 0
            elif this_itr > 20 and d_err>self.min_tolerance:
                log.info("\nToo many loops at this tolerance, stuck?")
                itr += 1
                self.sub_fitter.tolerance = self.sub_fitter.tolerance / 2
                this_itr = 0

            err = new_err

        # report that fit is complete and why stopped
        if itr > self.max_iter:
            log.info("\nIter {0}: max number of iterations completed"
                         .format(itr))
        if d_err<=self.min_tolerance:
            log.info("\nIter {0}: error improvement < min_tolerance"
                         .format(itr,self.min_tolerance))

        # Fit all params together aferward. If iterative fit did its job,
        # this should be a very short operation.
        # May only be useful for testing.
        log.info("Subfitting complete, beginning a final whole-model fit (DELETE ME?)")
        self.sub_fitter.fit_modules = self.fit_modules
        err = self.sub_fitter.do_fit()

        # These should match
        log.debug("self.stack.error() is: {0}\n"
                  "local err variable is: {1}\n"
                  .format(self.stack.error(), err))

        #return(self.stack.error())
        return err

'''

def fit_iteratively(
        data, modelspec,
        fitter=coordinate_descent,
        segmentor=nems.segmentors.use_all_data,
        mapper=nems.fitters.mappers.simple_vector,
        metric=lambda data: nems.metrics.api.nmse(data, 'pred', 'resp'),
        metaname='fit_basic', fit_kwargs={},
        module_sets=None, invert=False, tolerances=None,
        ):
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

     module_sets   A nested list specifying which model indices should be fit.
                   Overall iteration will occurr len(module_sets) many times.
                   ex: [[0], [1, 3], [0, 1, 2, 3]]

     invert        Boolean. Causes module_sets to specify the model indices
                   that should *not* be fit.


    Returns
    A list containing a single modelspec, which has the best parameters found
    by this fitter.
    '''
    if module_sets is None:
        module_sets = [[i] for i in range(len(modelspec))]

    if tolerances is None:
        tolerances = [1e-6]

    start_time = time.time()

    # Ensure that phi exists for all modules; choose prior mean if not found
    for i, m in enumerate(modelspec):
        if not m.get('phi'):
            log.info('Phi not found for module, using mean of prior: {}'
                      .format(m))
            m = nems.priors.set_mean_phi([m])[0]  # Inits phi for 1 module
            modelspec[i] = m

    # Create the mapper object that translates to and from modelspecs.
    # It has two methods that, when defined as mathematical functions, are:
    #    .pack(modelspec) -> fitspace_point
    #    .unpack(fitspace_point) -> modelspec
    packer, unpacker = mapper(modelspec)

    # A function to evaluate the modelspec on the data
    evaluator = ms.evaluate

    # get initial sigma value representing some point in the fit space
    sigma = packer(modelspec)

    for tol in tolerances:
        log.info("Fitting all subsets with tolerance: %.2E", tol)
        for subset in module_sets:
            log.info("Fitting subset: %s\n", subset)
            if invert:
                # invert the indices
                subset_inverted = [
                        None if i in subset else i
                        for i, in enumerate(modelspec)
                        ]
                subset = [i for i in subset_inverted if i is not None]
                log.debug("Inverted subset: %s\n", subset)

            # remove hold_outs from modelspec
            held_out = []
            subtracted_modelspec = []
            for i, m in enumerate(modelspec):
                if i in subset:
                    held_out.append(None)
                    subtracted_modelspec.append(m)
                else:
                    held_out.append(m)
            log.debug("\n\nheld_out subset was: %s\n\nsubtracted_modelspec: %s",
                      held_out, subtracted_modelspec)

            def cost_function(sigma, unpacker, modelspec, data,
                              evaluator, metric, held_out):
                updated_spec = unpacker(sigma)
                # The segmentor takes a subset of the data for fitting each step
                # Intended use is for CV or random selection of chunks of the data
                data_subset = segmentor(data)

                # Put hold_out back in before evaluating
                recombined_spec = []
                j = 0
                for i, m in enumerate(held_out):
                    if m is None:
                        recombined_spec.append(updated_spec[j])
                        j += 1
                    else:
                        recombined_spec.append(m)

                updated_data_subset = evaluator(data_subset, recombined_spec)
                error = metric(updated_data_subset)
                log.debug("inside cost function, current error: %.06f", error)
                log.debug("\ncurrent sigma: %s", sigma)

                cost_function.counter += 1
                if cost_function.counter % 1000 == 0:
                    log.info('Eval #%d. E=%.06f', cost_function.counter, error)
                    log.debug("\n\nrecombined_spec was: %s", recombined_spec)

                return error

            cost_function.counter = 0

            # Freeze everything but sigma, since that's all the fitter should be
            # updating.
            cost_fn = partial(cost_function,
                              unpacker=unpacker, modelspec=subtracted_modelspec,
                              data=data, evaluator=evaluator,
                              metric=metric, held_out=held_out)

            # do fit
            improved_sigma = fitter(sigma, cost_fn, tolerance=tol,
                                    **fit_kwargs)
            improved_modelspec = unpacker(improved_sigma)

            recombined_modelspec = [
                    phi if phi is not None else held_out[i]
                    for i, phi in enumerate(improved_modelspec)
                    ]
            log.debug("\n\nsubset: %s\nrecombined_modelspec: %s",
                      subset, recombined_modelspec)

    elapsed_time = (time.time() - start_time)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.set_modelspec_metadata(recombined_modelspec, 'fitter', metaname)
    ms.set_modelspec_metadata(recombined_modelspec, 'fit_time', elapsed_time)
    results = [copy.deepcopy(recombined_modelspec)]

    return results
