import logging
import copy

from nems.fitters.fitter import scipy_minimize
import nems.metrics.api as metrics
from .fit_basic import fit_basic
from .fit_iteratively import fit_iteratively

log = logging.getLogger(__name__)


def fit_nfold(data_list, modelspecs, generate_psth=False,
              fitter=scipy_minimize, analysis='fit_basic',
              metric=None, tolerances=None, module_sets=None,
              tol_iter=100, fit_iter=20, fit_kwargs={}):
    '''
    Takes njacks jackknifes, where each jackknife has some small
    fraction of data NaN'd out, and fits modelspec to them.

    TESTING:
    if input len(modelspecs) == len(data_list) then use each
      modelspec as initial condition for corresponding data_list fold
    if len(modelspecs) == 1, then use the same initial conditions for
      each fold

    '''
    # fit_kwargs = fit_kwargs.copy()
    # if 'options' not in fit_kwargs.keys():
    #     fit_kwargs['options'] = {}
    # if 'ftol' not in fit_kwargs['options'].keys():
    #     fit_kwargs['options']['ftol'] = 1e-7
    # if 'maxiter' not in fit_kwargs['options'].keys():
    #     fit_kwargs['options']['maxiter'] = 1000

    nfolds = len(data_list)
    models = []
    if metric is None:
        metric = lambda d: metrics.nmse(d, 'pred', 'resp')

    for i in range(nfolds):
        if len(modelspecs) > 1:
            msidx = i
        else:
            msidx = 0
        log.info("Fitting fold %d/%d, modelspec %d", i+1, nfolds, msidx)
#        resp = data_list[i]['resp']
#        resp_len = np.sum(np.isfinite(resp.as_continuous()))
#        log.info("non-nan resp samples: %d", resp_len)
#        stim = data_list[i]['psth']
#        stim_len = np.sum(np.isfinite(stim.as_continuous()[0, :]))
#        log.info("non-nan stim samples: %d", stim_len)
        if analysis == 'fit_basic':
            models += fit_basic(data_list[i], copy.deepcopy(modelspecs[msidx]),
                                fitter=fitter,
                                metric=metric,
                                metaname='fit_nfold',
                                fit_kwargs=fit_kwargs)
        elif analysis == 'fit_iteratively':
            models += fit_iteratively(
                        data_list[i], copy.deepcopy(modelspecs[msidx]),
                        fitter=fitter, metric=metric,
                        metaname='fit_nfold', fit_kwargs=fit_kwargs,
                        module_sets=module_sets, invert=False,
                        tolerances=tolerances, tol_iter=tol_iter,
                        fit_iter=fit_iter,
                        )

        else:
            # Unknown analysis
            # TODO: use getattr / import to make this more general for
            #       use with any analysis function?
            #       Maybe too much of a pain.
            raise NotImplementedError

    return models
