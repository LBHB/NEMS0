import logging

import copy
import numpy as np

from nems.utils import split_keywords
from nems.registry import KeywordRegistry
from nems.plugins.keywords import default_keywords
from nems.fitters.api import scipy_minimize
import nems.priors
import nems.modelspec as ms
import nems.metrics.api as metrics

log = logging.getLogger(__name__)
default_kws = KeywordRegistry().register_module(default_keywords)


def from_keywords(keyword_string, registry=None, meta={}):
    '''
    Returns a modelspec created by splitting keyword_string on underscores
    and replacing each keyword with what is found in the nems.keywords.defaults
    registry. You may provide your own keyword registry using the
    registry={...} argument.
    '''
    if registry is None:
        registry = default_kws
    keywords = split_keywords(keyword_string)

    # Lookup the modelspec fragments in the registry
    modelspec = []
    for kw in keywords:
        if registry.kw_head(kw) not in registry:
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


def from_keywords_as_list(keyword_string, registry=None, meta={}):
    '''
    wrapper for from_keywords that returns modelspec as a modelspecs list,
    ie, [modelspec]
    '''
    if registry is None:
        registry = default_kws
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
            # log.info('exluding '+fn)
            # log.info(m['fn'])
            # log.info(m.get('phi'))
            if (fn in m['fn']):
                if (m.get('phi') is None):
                    m = nems.priors.set_mean_phi([m])[0]  # Inits phi
                    log.info('Mod %d (%s) fixing phi to prior mean', i, fn)
                else:
                    log.info('Mod %d (%s) fixing phi', i, fn)

                m['fn_kwargs'].update(m['phi'])
                m['phi'] = {}
                exclude_idx.append(i)
                # log.info(m)

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
            log.info('Intializing phi for module %d (%s)', i, m['fn'])
            m = nems.priors.set_mean_phi([m])[0]  # Inits phi

        log.info('Freezing phi for module %d (%s)', i, m['fn'])

        m['fn_kwargs'].update(m['phi'])
        m['phi'] = {}
        tmodelspec[i] = m

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
    target_i = _find_module('double_exponential', modelspec)
    if target_i is None:
        log.warning("No dexp module was found, can't initialize.")
        return modelspec

    if target_i == len(modelspec):
        fit_portion = modelspec
    else:
        fit_portion = modelspec[:target_i]

    # generate prediction from module preceeding dexp
    ms.fit_mode_on(fit_portion)
    rec = ms.evaluate(rec, fit_portion)
    ms.fit_mode_off(fit_portion)
    resp = rec['resp'].as_continuous()
    pred = rec['pred'].as_continuous()
    keepidx = np.isfinite(resp) * np.isfinite(pred)
    resp = resp[keepidx]
    pred = pred[keepidx]

    # choose phi s.t. dexp starts as almost a straight line
    # phi=[max_out min_out slope mean_in]
    # meanr = np.nanmean(resp)
    stdr = np.nanstd(resp)
    # base = np.max(np.array([meanr - stdr * 4, 0]))
    # amp = np.max(resp) - np.min(resp)
    base = np.min(resp)
    amp = stdr * 3
    # base = meanr - stdr * 3

    shift = np.mean(pred)
    # shift = (np.max(pred) + np.min(pred)) / 2
    predrange = 2 / (np.max(pred) - np.min(pred) + 1)

    modelspec[target_i]['phi'] = {'amplitude': amp, 'base': base,
                                  'kappa': np.log(predrange), 'shift': shift}
    log.info("Init dexp (amp,base,kappa,shift)=(%.3f,%.3f,%.3f,%.3f)",
             *modelspec[target_i]['phi'].values())

    return modelspec


def init_logsig(rec, modelspec):
    '''
    Initialization of priors for logistic_sigmoid,
    based on process described in methods of Rabinowitz et al. 2014.
    '''
    logsig_idx = _find_module('logistic_sigmoid', modelspec)
    if logsig_idx is None:
        log.warning("No logsig module was found, can't initialize.")
        return modelspec

    stim = rec['stim'].as_continuous()
    resp = rec['resp'].as_continuous()
    # TODO: Maybe need a more sophisticated calculation for this?
    #       Paper isn't very clear on how they calculate "X-bar" and "Y-bar"
    #       They also mention that their stim-resp data is split up into 20
    #       bins, maybe averaged across trials or something?
    mean_stim = np.nanmean(stim)
    min_stim = np.nanmin(stim)
    max_stim = np.nanmax(stim)
    stim_range = max_stim - min_stim
    min_resp = np.nanmin(resp)
    max_resp = np.nanmax(resp)
    resp_range = max_resp - min_resp

    # Rather than setting a hard value for initial phi,
    # set the prior distributions and let the fitter/analysis
    # decide how to use it.
    base = ('Exponential', {'beta': min_resp + 0.05*(resp_range)})
    amplitude = ('Exponential', {'beta': 2*resp_range})
    shift = ('Normal', {'mean': mean_stim, 'sd': stim_range})
    kappa = ('Exponential', {'beta': stim_range/stim.shape[1]})

    modelspec[logsig_idx]['prior'] = {
            'base': base, 'amplitude': amplitude, 'shift': shift,
            'kappa': kappa
            }
    log.info("logistic_sigmoid priors initialized to: "
             "base: %s\namplitude: %s\nshift: %s\nkappa: %s\n",
             *modelspec[logsig_idx]['prior'].values())

    return modelspec


def _find_module(name, modelspec):
    target_i = None
    target_module = name
    for i, m in enumerate(modelspec):
        if target_module in m['fn']:
            target_i = i
            break

    if not target_i:
        log.info("target_module: %s not found in modelspec.", target_module)

    log.info("target_module: %s found at modelspec[%d]",
             target_module, target_i)

    return target_i
