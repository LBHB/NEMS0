# Fitters

To keep fitters as simple and re-useable as possible, they should only require `sigma` (some representatio of a point in fit-space, like a vector) and `cost_fn` (some function for determining error based on sigma) as arguments. This keeps fitter functions from needing to know anything about the model architecture, and ensures easy compatibility with popular fitting packages like scipy.minimize.
Any additional functionality related to fitting, such as only fitting certain subsets of data or using a different fitter for a specific module, should be handled by an analysis (for a generalizeable function) or a script (for a very specific problem) as appropriate.

## Termination Conditions

The termination conditions fitters are usually pretty similar, so we placed some generic termination condition functions in `fitters/termination_conditions.py`. Most fitters will have similar conditions for when to stop the fitting process: when the elapsed time, step size, or error delta reaches some value.

Termination condition functions should take in a step_info dict and return a boolean. If more than one condition should be checked, fitters may combine more than one termination conditions using the appropriate `and` or `or` expressions and a lambda function.

For example, many fitters look like:

```
import termination_conditions as tc

def my_fitter(sigma, cost_fn):

    stepinfo, update_stepinfo = tc.create_stepinfo()
    stop_fit = lambda : (tc.error_non_decreasing(stepinfo, tolerance) or
                         tc.max_iterations_reached(stepinfo, 1000))

    while not stop_fit():
        better_sigma = ...     # Find a better sigma somehow
        sigma = better_sigma
        err = cost_fn(sigma=sigma)
        update_stepinfo(err=err, sigma=sigma)

    return sigma
```

### Conventions for Termination Conditions

1) The name of the function should describe the event that will cause the
   fitting loop to stop, and return True to indicate a stop. For example,
   'error_non_decreasing' returns a value of True when, as the name implies,
   the error is no longer decreasing by an amount at least equal to the
   specified tolerance.

   As a result, fitters should generally refer to the step_condition
   in terms of: 'if termination_condition <is True>:  <stop fitting>'
   Using the reverse naming and return values should be avoided.

2) The expected structure of step_info is made with create_stepinfo().
   For example:

       stepinfo = {'num': 1,             # Num of steps taken thus far
                   'err': 03.93,         # The cost to minimize
                   'err_delta': None,    # Change in cost since last step
                   'start_time': time.time()}

   These are the four default keys, but you may add more if you want to have more exotic termination conditions.
