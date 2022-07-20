import logging
import time
import numpy as np

log = logging.getLogger(__name__)

# Reusable, composable functions for determining when you should stop
# fitting. Use of this code is optional (fitters may always do as they wish)
# but recommended unless you have an unusual fitting algorithm.


def create_stepinfo():
    '''
    Returns (stepinfo, stepinfo_fn), where
       stepinfo          Default dict used with all termination conditions
       update_stepinfo   Function of single argument (err) that lets you
                         update stepinfo after you take each step. Use of
                         this function is optional but recommended.

    Example use:
       import termination_conditions as tc

       stepinfo, update_stepinfo = tc.create_stepinfo()
       stop_fit = lambda : (tc.error_non_decreasing(stepinfo, tolerance) or
                            tc.max_iterations_reached(stepinfo, 1000))

       while not stop_fit():
           # ... mutate sigma as appropriate here
           err = cost_fn(sigma=sigma)
           update_stepinfo(err=err, sigma=sigma)

    The stepinfo data structure is to be used with all the termination
    condition functions defined in this file.
    '''
    stepinfo = {
            'stepnum': 0,
            'err': np.inf,
            'err_delta': -np.inf,
            'start_time': time.time()
            }

    def update_stepinfo(err, **kwargs):
        stepinfo['stepnum'] += 1
        stepinfo['err_delta'] = err - stepinfo['err']
        stepinfo['err'] = err
        stepinfo.update(kwargs)
        log.debug("Stepinfo: %s", stepinfo)

    return stepinfo, update_stepinfo


def error_non_decreasing(stepinfo, tolerance=1e-5):
    '''
    Returns true when stepinfo's 'err_delta' is less than tolerance.
    The default tolerance is 1.0e-5.
    '''
    if (stepinfo['err_delta'] > -tolerance) and (stepinfo['err_delta'] < 0):
        log.info("Change in error: %.06f was less than tolerance: %.2E",
                 stepinfo['err_delta'], tolerance)
        return True
    else:
        return False


def fit_time_exceeded(stepinfo, max_time=300):
    '''
    Returns True when stepinfo's 'start_time' is at least max_time seconds ago.
    The default max_time is 300 seconds (5 minutes)
    '''
    # time.time() and stepinfo['start_time'] are in units of seconds
    if (time.time() - stepinfo['start_time']) >= max_time:
        log.info("Maximum fit time exceeded: %.02f", max_time)
        return True
    else:
        return False


def max_iterations_reached(stepinfo, max_iter=10000):
    '''
    Returns true when stepinfo's 'stepnum' is greater than or equal to
    max_iter. The default of max_iter is 10,000.
    '''
    if stepinfo['stepnum'] >= max_iter:
        log.info("Maximum iterations reached: %d", max_iter)
        return True
    else:
        return False


def target_err_reached(stepinfo, target=0.0):
    '''
    Returns true when 'err' is less than or equal to target.
    The default target is 0.0
    '''
    if stepinfo['err'] <= target:
        log.info("Target error reached: %.06f", target)
        return True
    else:
        return False

def less_than_equal(a, b):
    '''
    Returns true when a is less than or equal to b.
    Generic condition for fitters that want to check conditions
    specific to that function, but want to keep formatting consistent.

    Note: This is not a smart function, it is only a wrapper for
          a simple <= expression exactly as it is evalualted by
          Python by default.
    '''
    if a <= b:
        return True
    else:
        return False