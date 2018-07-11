import logging
import copy

import nems.priors
from nems.analysis.fit_basic import fit_basic
from nems.analysis.fit_iteratively import fit_iteratively

log = logging.getLogger(__name__)


def fit_from_priors(data, modelspec, ntimes=10, analysis='fit_basic',
                    basic_kwargs={}, iter_kwargs={}):
    '''
    Fit ntimes times, starting from random points sampled from the prior.

    TODO : Test, add more parameters
    '''
    models = []
    for i in range(ntimes):
        log.info("Fitting from random start: {}/{}".format(i+1, ntimes))
        ms = nems.priors.set_random_phi(copy.deepcopy(modelspec))

        if analysis == 'fit_basic':
            models.append(fit_basic(data, ms, metaname='fit_from_priors',
                                    **basic_kwargs)[0])
        elif analysis == 'fit_iteratively':
            models.append(fit_iteratively(data, ms, metaname='fit_from_priors',
                                          **iter_kwargs)[0])
        else:
            raise NotImplementedError("No support for analysis: %s" % analysis)

    return models
