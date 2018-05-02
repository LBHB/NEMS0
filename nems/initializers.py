import logging

import copy
import numpy as np

from nems.utils import split_keywords
from nems import keywords
from nems.fitters.api import scipy_minimize
import nems.priors
import nems.modelspec as ms
import nems.metrics.api as metrics

log = logging.getLogger(__name__)


def from_keywords(keyword_string, registry=keywords.defaults, meta={}):
    '''
    Returns a modelspec created by splitting keyword_string on underscores
    and replacing each keyword with what is found in the nems.keywords.defaults
    registry. You may provide your own keyword registry using the
    registry={...} argument.
    '''
    keywords = split_keywords(keyword_string)

    # Lookup the modelspec fragments in the registry
    modelspec = []
    for kw in keywords:
        if kw not in registry:
            raise ValueError("unknown keyword: {}".format(kw))
        d = copy.deepcopy(registry[kw])
        d['id'] = kw
        modelspec.append(d)

    # first module that takes input='pred' should take 'stim' instead.
    # can't hard code in keywords, since we don't know which keyword will be
    # first. and can't assume that it will be module[0] because those might be
    # state manipulations
    first_input_to_stim = False
    i = 0
    while not first_input_to_stim and i < len(modelspec):
        if 'i' in modelspec[i]['fn_kwargs'].keys() and \
           modelspec[i]['fn_kwargs']['i'] == 'pred':
            if 'state' in modelspec[i]['fn']:
                modelspec[i]['fn_kwargs']['i'] = 'psth'
            else:
                modelspec[i]['fn_kwargs']['i'] = 'stim'
            first_input_to_stim = True
        i += 1

    # insert metadata, if provided
    if 'meta' not in modelspec[0].keys():
        modelspec[0]['meta'] = meta
    else:
        modelspec[0]['meta'].update(meta)

    return modelspec


def from_keywords_as_list(keyword_string, registry=keywords.defaults, meta={}):
    '''
    wrapper for from_keywords that returns modelspec as a modelspecs list,
    ie, [modelspec]
    '''
    return [from_keywords(keyword_string, registry, meta)]


def prefit_to_target(rec, modelspec, analysis_function, target_module,
                     extra_exclude=[],
                     fitter=scipy_minimize, fit_kwargs={}):
    """Removes all modules from the modelspec that come after the
    first occurrence of the target module, then performs a
    rough fit on the shortened modelspec, then adds the latter
    modules back on and returns the full modelspec.
    """

    # figure out last modelspec module to fit
    target_i = None
    for i, m in enumerate(modelspec):
        if target_module in m['fn']:
            target_i = i + 1
            break

    if not target_i:
        log.info("target_module: {} not found in modelspec."
                 .format(target_module))
        return modelspec
    else:
        log.info("target_module: {0} found at modelspec[{1}]."
                 .format(target_module, target_i-1))

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    exclude_idx = []
    tmodelspec = []
    for i in range(target_i):
        m = copy.deepcopy(modelspec[i])
        for fn in extra_exclude:
#            log.info('exluding '+fn)
#            log.info(m['fn'])
#            log.info(m.get('phi'))
            if (fn in m['fn']):
                if (m.get('phi') is None):
                   m = nems.priors.set_mean_phi([m])[0]  # Inits phi
                   log.info('Mod %d (%s) fixing phi to prior mean', i, fn)
                else:
                   log.info('Mod %d (%s) fixing phi', i, fn)

                m['fn_kwargs'].update(m['phi'])
                del m['phi']
                exclude_idx.append(i)

        tmodelspec.append(m)

    # fit the subset of modules
    tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                   fit_kwargs=fit_kwargs)[0]

    # reassemble the full modelspec with updated phi values from tmodelspec
    for i in np.setdiff1d(np.arange(target_i), np.array(exclude_idx)):
        modelspec[i] = tmodelspec[i]

    return modelspec


def prefit_mod_subset(rec, modelspec, analysis_function,
                      fit_set=[],
                      fitter=scipy_minimize, fit_kwargs={}):
    """Removes all modules from the modelspec that come after the
    first occurrence of the target module, then performs a
    rough fit on the shortened modelspec, then adds the latter
    modules back on and returns the full modelspec.
    """

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    fit_idx = []
    tmodelspec = []
    for i, m in enumerate(modelspec):
        m = copy.deepcopy(m)
        for fn in fit_set:
            if fn in m['fn']:
                fit_idx.append(i)
                log.info('Found module %d (%s) for subset prefit', i, fn)
        tmodelspec.append(m)

    if len(fit_idx) == 0:
        log.info('No modules matching fit_set for subset prefit')
        return modelspec

    exclude_idx = np.setdiff1d(np.arange(0, len(modelspec)),
                               np.array(fit_idx))
    for i in exclude_idx:
        m = tmodelspec[i]
        if not m.get('phi'):
            m = nems.priors.set_mean_phi([m])[0]  # Inits phi

        log.debug('Freezing phi for module %d (%s)', i, m['fn'])

        m['fn_kwargs'].update(m['phi'])
        m['phi'] = {}
        # tmodelspec[i] = m

    # fit the subset of modules
    tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                   fit_kwargs=fit_kwargs)[0]

    # reassemble the full modelspec with updated phi values from tmodelspec
    for i in fit_idx:
        modelspec[i] = tmodelspec[i]

    return modelspec


def init_dexp(rec, modelspec):
    """
    choose initial values for dexp applied after preceeding fir is
    initialized
    """
    target_i = None
    target_module = 'double_exponential'
    for i, m in enumerate(modelspec):
        if target_module in m['fn']:
            target_i = i
            break

    if not target_i:
        log.info("target_module: %s not found in modelspec.", target_module)
        return modelspec

    log.info("target_module: %s found at modelspec[%d]",
             target_module, target_i)

    if target_i == len(modelspec):
        fit_portion = modelspec
    else:
        fit_portion = modelspec[:target_i]

    # generate prediction from module preceeding dexp
    rec = ms.evaluate(rec, fit_portion)
    resp = rec['resp'].as_continuous()
    pred = rec['pred'].as_continuous()
    keepidx = np.isfinite(resp) * np.isfinite(pred)
    resp = resp[keepidx]
    pred = pred[keepidx]

    # choose phi s.t. dexp starts as almost a straight line
    # phi=[max_out min_out slope mean_in]
    meanr = np.nanmean(resp)
    stdr = np.nanstd(resp)
    # base = np.max(np.array([meanr - stdr * 4, 0]))
    # amp = np.max(resp) - np.min(resp)
    base = np.min(resp)
    amp = stdr * 3
    # base = meanr - stdr * 3

    shift = np.mean(pred)
    #shift = (np.max(pred) + np.min(pred)) / 2
    predrange = 2 / (np.max(pred) - np.min(pred) + 1)

    modelspec[target_i]['phi'] = {}
    modelspec[target_i]['phi']['amplitude'] = amp
    modelspec[target_i]['phi']['base'] = base
    modelspec[target_i]['phi']['kappa'] = np.log(predrange)
    modelspec[target_i]['phi']['shift'] = shift
    log.info("Init dexp (amp,base,kappa,shift)=(%.3f,%.3f,%.3f,%.3f)",
             modelspec[target_i]['phi']['amplitude'],
             modelspec[target_i]['phi']['base'],
             modelspec[target_i]['phi']['kappa'],
             modelspec[target_i]['phi']['shift'])

#    rec = ms.evaluate(rec, modelspec)
#    x = rec['resp'].as_continuous()
#    y = rec['pred'].as_continuous()
#    keepidx = np.isfinite(x) * np.isfinite(y)
#    x = x[keepidx]
#    y = y[keepidx]
#

    return modelspec
