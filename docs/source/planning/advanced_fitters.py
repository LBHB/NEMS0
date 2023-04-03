# ----------------------------------------------------------------------------
# INTERLUDE: Giving names to fitter internal functions
#
# We are about to get into more difficult territory, and so once again I am
# going to make some mathematical definitions that may be useful as
# optional arguments (or internal functions) used in our fitters.
#
# |---------------------------+------------------------------------------------|
# | f(modelspec) -> error     | COST_FN. (Closed over function references EST) |
# | f(stepinfo) -> boolean    | STOP_COND. Decides when to stop the fit        |
# | f(modelspec) -> stepspace | PACKER. Converts a modelspec into a stepspace  |
# | f(stepspace) -> modelspec | UNPACKER. The inverse operation of PACKER      |
# | f(stepspace) -> stepspace | STEPPER. Alters 1+ parameters in stepspace     |
#
# I used to think that "packers" and "unpackers" should always convert to
# vectors, combining all params into "phi_all". However, now I think it may be
# better to consider them as generating a point in "stepspace" in which the
# fitting algorithm may take a step, and leave the nature of this subspace
# vague -- it may not be a vector space, and the number of dimensions is not
# fixed.
#
# My rationale for this is that:
#
# 1) I can think of certain fitting situations in which the modelspec should not
#    become a parameter vector (such as during a bayesian fit, where the
#    modelspec gives a distribution rather than a single parameter);
#
# 2) We may want to develop different "packers" and "unpackers" that strive to
#    keep the parameter search space (the "stepspace") linear, even if the
#    parameters themselves create a fairly nonlinear space.
#
# 3) By not assuming anything about the stepspace, we can easily change its
#    dimensionality from step to step by creating a triplet of functions on
#    every step that pack, step, and unpack themselves.

# ----------------------------------------------------------------------------
# Example: fitting iteratively, giving every module a different fitter

# Everything is as before, except for the one line that runs the fitter is
# replaced with the following:

import time
import skopt
from nems0.termination_condition import error_nondecreasing

per_module_fitters = {'nems.modules.fir': skopt.gp_minimize,
                      'nems.modules.nl/dexp': nems.fitters.take_greedy_10_steps}

def iterative_permodule_fitter(fitterdict, modelspec, cost_fn,
                               stop_cond=error_nondecreasing):
    '''Fit each module, one at a time'''
    spec = modelspec
    stepinfo = {'num': 1, # Num of steps
                'err': cost_fn(spec), #
                'err_delta': None, # Change since last step
                'start_time': time.time()}
    # Check for errors
    for m in modelspec:
        if m['fn'] not in fitterdict:
            raise ValueError('Missing per module fitter:'+m['fn'])
    # Now do the actual iterated fit
    while error_nondecreasing(stepinfo):
        for m in modelspec:
            m['phi'] = fitterdict[m['fn']](m['phi'])
        stepinfo = update_stepinfo() # TODO
    return spec

# Build the fitter fn and do the fitting process
from functools import partial
fitter = partial(iterative_permodule_fitter, per_module_fitters)
modelspec_fitted = fitter(modelspec, est_evaluator)


# ----------------------------------------------------------------------------
# Even harder example: fitting different random subsets of the parameters
# on each fitter search round

def random_subset_fitter(modelspec, cost_fn,
                         stop_cond=...):
    cost_fn = lambda spec: metric(evaluator(spec))
    spec = modelspec
    while not stop_cond(stepinfo):
        (packer, stepper, unpacker) = make_random_subset_triplet()  # TODO
        stepspace_point = packer(modelspec)
        best_spec = stepper(stepspace_point, unpacker, cost_fn)
        spec = unpacker(best_spec)
    return spec


# ----------------------------------------------------------------------------
# I think you can see how you could have fitters:
# - Return multiple parameter sets, such as in a bayesian analysis or swarm model
# - Completely drop out entire modules or parameters (because the modules are
#      not really in existance; they are just part of the modelspec datastructure)
# - Return the whole set of modelspecs trained on the jackknifed est data
# etc

