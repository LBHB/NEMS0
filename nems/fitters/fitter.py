import logging
import numpy as np
import scipy as scp

import nems.fitters.termination_conditions as tc

log = logging.getLogger(__name__)


def dummy_fitter(sigma, cost_fn, bounds=None, fixed=None):
    '''
    This fitter does not actually take meaningful steps; it merely
    varies the first element of the sigma vector to be equal to the step
    number. It is intended purely for testing and example purposes so
    that you can see how to re-use termination conditions to write
    your own fitter.
    '''
    # Define a stepinfo and termination condition function 'stop_fit'
    stepinfo, update_stepinfo = tc.create_stepinfo()
    stop_fit = lambda : (tc.error_non_decreasing(stepinfo, 1e-5) or
                         tc.max_iterations_reached(stepinfo, 1000))

    while not stop_fit():
        sigma[0] = stepinfo['stepnum']  # Take a fake step
        err = cost_fn(sigma)
        update_stepinfo(err=err, sigma=sigma)

    return sigma


def coordinate_descent(sigma, cost_fn, step_size=0.1, step_change=0.5,
                       step_min=1e-5, tolerance=1e-2, max_iter=100, **kwargs):

    stepinfo, update_stepinfo = tc.create_stepinfo()
    stop_fit = lambda : (tc.error_non_decreasing(stepinfo, tolerance)
                         or tc.max_iterations_reached(stepinfo, max_iter)
                         or tc.less_than_equal(step_size, step_min))

    this_sigma = sigma.copy()
    n_parameters = len(sigma)
    step_errors = np.empty([n_parameters, 2])
    while not stop_fit():
        for i in range(0, n_parameters):
            # Try shifting each parameter both negatively and positively
            # proportional to step_size, and save both the new
            # sigma vectors and resulting cost_fn outputs
            this_sigma[i] = sigma[i] + step_size
            step_errors[i, 0] = cost_fn(this_sigma)
            this_sigma[i] = sigma[i] - step_size
            step_errors[i, 1] = cost_fn(this_sigma)
            this_sigma[i] = sigma[i]
        # Get index tuple for the lowest error that resulted,
        # and keep the corresponding sigma vector for the next iteration
        i_param, j_sign = np.unravel_index(
                                step_errors.argmin(), step_errors.shape
                                )
        err = step_errors[i_param, j_sign]

        # If change was negative, try reducing step size.
        if err > stepinfo['err']:
            log.debug("Error got worse, reducing step size from: %.06f to: "
                      "%.06f", step_size, step_size * step_change)
            step_size *= step_change
            continue

        # If j is 1, shift was negative, otherwise it was 0 for positive.
        if j_sign == 1:
            sigma[i_param] = this_sigma[i_param] = sigma[i_param] - step_size
        else:
            sigma[i_param] = this_sigma[i_param] = sigma[i_param] + step_size

        update_stepinfo(err=err)

        if stepinfo['stepnum'] % 20 == 0:
            log.debug("sigma is now: %s", sigma)

    log.info("Final error: %.06f\n", stepinfo['err'])
    return sigma


def scipy_minimize(sigma, cost_fn, tolerance=None, max_iter=None,
                   method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-7}):
    """
    Wrapper for scipy.optimize.minimize to normalize format with
    NEMS fitters.

    TODO: finish this doc

    Does not currently use the stepinfo/termination_conditions
    paradigm.

    TODO: Pull in code from scipy.py in docs/planning to
          expose more output during iteration.
    """
    if tolerance is not None:
        if 'ftol' in options:
            log.warn("Both <tolerance> and <options: ftol> provided for\n"
                     "scipy_minimize, using <tolerance> by default: %.2E",
                     tolerance)
        options['ftol'] = tolerance

    if max_iter is not None:
        if 'maxiter' in options:
            log.warn("Both <max_iter> and <options: maxiter> provided for\n"
                     "scipy_minimize, using <max_iter> by default: %d",
                     max_iter)
        options['maxiter'] = max_iter

    result = scp.optimize.minimize(cost_fn, sigma, method=method,
                                   options=options)
    sigma = result.x
    final_err = cost_fn(sigma)
    log.info("Final error: %.06f\n", final_err)
    return sigma
