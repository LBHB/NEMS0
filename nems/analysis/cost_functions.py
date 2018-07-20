import logging

import nems.utils

log = logging.getLogger(__name__)


def basic_cost(sigma, unpacker, modelspec, data, segmentor,
               evaluator, metric):
    '''Standard cost function for use by fit_basic and other analyses.'''
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
