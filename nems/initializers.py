import logging

import copy
import numpy as np

from nems.registry import KeywordRegistry
from nems.plugins import default_keywords
from nems.utils import find_module
from nems.analysis.api import fit_basic
from nems.fitters.api import scipy_minimize
import nems.priors as priors
import nems.modelspec as ms
import nems.metrics.api as metrics
from nems import get_setting

log = logging.getLogger(__name__)
default_kws = KeywordRegistry()
default_kws.register_module(default_keywords)
default_kws.register_plugins(get_setting('KEYWORD_PLUGINS'))


def from_keywords(keyword_string, registry=None, rec=None, meta={}, init_phi_to_mean_prior=True):
    '''
    Returns a modelspec created by splitting keyword_string on underscores
    and replacing each keyword with what is found in the nems.keywords.defaults
    registry. You may provide your own keyword registry using the
    registry={...} argument.
    '''
    if registry is None:
        registry = default_kws
    keywords = keyword_string.split('-')

    # Lookup the modelspec fragments in the registry
    modelspec = ms.ModelSpec()
    for kw in keywords:
        if kw.startswith("fir.Nx") and (rec is not None):
            N = rec['stim'].nchans
            kw_old = kw
            kw = kw.replace("fir.N", "fir.{}".format(N))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)
        elif kw.startswith("stategain.N") and (rec is not None):
            N = rec['state'].nchans
            kw_old = kw
            kw = kw.replace("stategain.N", "stategain.{}".format(N))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)
        elif (kw.endswith(".S") or ".Sx" in kw) and (rec is not None):
            S = rec['state'].nchans
            kw_old = kw
            kw = kw.replace(".S", ".{}".format(S))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)

        elif kw.endswith(".N") and (rec is not None):
            N = rec['stim'].nchans
            kw_old = kw
            kw = kw.replace(".N", ".{}".format(N))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)

        elif kw.endswith(".R") and (rec is not None):
            R = rec['resp'].nchans
            kw_old = kw
            kw = kw.replace(".R", ".{}".format(R))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)

        elif kw.endswith("xR") and (rec is not None):
            R = rec['resp'].nchans
            kw_old = kw
            kw = kw.replace("xR", "x{}".format(R))
            log.info("kw: dynamically subbing %s with %s", kw_old, kw)

        else:
            log.info('kw: %s', kw)
        if registry.kw_head(kw) not in registry:
            raise ValueError("unknown keyword: {}".format(kw))

        d = copy.deepcopy(registry[kw])
        d['id'] = kw
        d = priors.set_mean_phi([d])[0]  # Inits phi for 1 module

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
    if rec is not None:
        if ((rec['resp'].shape[0] > 1) and ('cellids' not in meta.keys()) and
            (type(rec.meta['cellid']) is list)):
            meta['cellids'] = rec.meta['cellid']

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


def prefit_LN(est, modelspec, analysis_function=fit_basic,
              fitter=scipy_minimize, metric=None, norm_fir=False,
              tolerance=10**-5.5, max_iter=700, nl_mode=2):
    '''
    Initialize modelspecs in a way that avoids getting stuck in
    local minima.

    written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)

    input: a single est recording and a single modelspec

    output: a single modelspec

    TODO -- make sure this works generally or create alternatives

    '''

    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    # Instead of using FIR prior, initialize to random coefficients then
    # divide by L2 norm to force sum of squares = 1
    if norm_fir:
        modelspec = fir_L2_norm(modelspec)

    # fit without STP module first (if there is one)
    modelspec = prefit_to_target(est, modelspec, fit_basic,
                                 target_module='levelshift',
                                 extra_exclude=['stp'],
                                 fitter=fitter,
                                 metric=metric,
                                 fit_kwargs=fit_kwargs)

    # then initialize the STP module (if there is one)
    for i, m in enumerate(modelspec):
        if 'stp' in m['fn']:
            m = priors.set_mean_phi([m])[0]  # Init phi for module
            modelspec[i] = m
            break

    # pre-fit static NL if it exists
    for m in modelspec:
        if 'double_exponential' in m['fn']:
            modelspec = init_dexp(est, modelspec, nl_mode=nl_mode)
            modelspec = prefit_mod_subset(
                    est, modelspec, fit_basic,
                    fit_set=['double_exponential'],
                    fitter=scipy_minimize,
                    metric=metric,
                    fit_kwargs=fit_kwargs)
            break

        elif 'logistic_sigmoid' in m['fn']:
            log.info("initializing priors and bounds for logsig ...\n")
            modelspec = init_logsig(est, modelspec)
            modelspec = prefit_mod_subset(
                    est, modelspec, fit_basic,
                    fit_set=['logistic_sigmoid'],
                    fitter=scipy_minimize,
                    metric=metric,
                    fit_kwargs=fit_kwargs)
#            for i, m in enumerate(modelspec):
#                if ('phi' not in m.keys()) and ('prior' in m.keys()):
#                    log.debug('Phi not found for module, using mean of prior: %s',
#                              m)
#                    old_prior = m['prior'].copy()
#                    m = priors.set_mean_phi([m])[0]  # Inits phi for 1 module
#                    modelspec[i] = m
#                    modelspec[i]['prior'] = old_prior
            break

#                modelspecs = [prefit_to_target(
#                        est, modelspec, fit_basic,
#                        target_module='double_exponential',
#                        extra_exclude=['stp'],
#                        fitter=scipy_minimize,
#                        fit_kwargs={'tolerance': 1e-6, 'max_iter': 500})
#                        for modelspec in modelspecs]

    return modelspec


def prefit_to_target(rec, modelspec, analysis_function, target_module,
                     extra_exclude=[],
                     fitter=scipy_minimize, metric=None,
                     fit_kwargs={}):
    """Removes all modules from the modelspec that come after the
    first occurrence of the target module, then performs a
    rough fit on the shortened modelspec, then adds the latter
    modules back on and returns the full modelspec.
    """

    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

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
    tmodelspec = ms.ModelSpec()
    for i in range(len(modelspec)):
        m = copy.deepcopy(modelspec[i])
        for fn in extra_exclude:
            # log.info('exluding '+fn)
            # log.info(m['fn'])
            # log.info(m.get('phi'))
            if (fn in m['fn']):
                if (m.get('phi') is None):
                    m = priors.set_mean_phi([m])[0]  # Inits phi
                    log.info('Mod %d (%s) fixing phi to prior mean', i, fn)
                else:
                    log.info('Mod %d (%s) fixing phi', i, fn)

                m['fn_kwargs'].update(m['phi'])
                m['phi'] = {}
                exclude_idx.append(i)
                # log.info(m)

        if ('levelshift' in m['fn']):
            m = priors.set_mean_phi([m])[0]
            try:
                mean_resp = np.nanmean(rec['resp'].as_continuous(), axis=1, keepdims=True)
            except NotImplementedError:
                # as_continous only available for RasterizedSignal
                mean_resp = np.nanmean(rec['resp'].rasterize().as_continuous(), axis=1, keepdims=True)
            log.info('Mod %d (%s) fixing level to response mean %.3f',
                     i, m['fn'], mean_resp[0])
            log.info('resp has %d channels', len(mean_resp))
            m['phi']['level'][:] = mean_resp

        if (i < target_i) or ('merge_channels' in m['fn']):
            tmodelspec.append(m)

    # fit the subset of modules
    if metric is None:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       fit_kwargs=fit_kwargs)
    else:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       metric=metric, fit_kwargs=fit_kwargs)
    if type(tmodelspec) is list:
        # backward compatibility
        tmodelspec = tmodelspec[0]

    # reassemble the full modelspec with updated phi values from tmodelspec
    #print(modelspec[0])
    #print(tmodelspec[0])
    for i in np.setdiff1d(np.arange(target_i), np.array(exclude_idx)).tolist():
        modelspec[int(i)] = tmodelspec[int(i)]

    return modelspec


def prefit_mod_subset(rec, modelspec, analysis_function,
                      fit_set=[],
                      fitter=scipy_minimize, metric=None, fit_kwargs={}):
    """Removes all modules from the modelspec that come after the
    first occurrence of the target module, then performs a
    rough fit on the shortened modelspec, then adds the latter
    modules back on and returns the full modelspec.
    """

    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
#    fit_idx = []
#    tmodelspec = []
#    for i, m in enumerate(modelspec):
#        m = copy.deepcopy(m)
#        for fn in fit_set:
#            if fn in m['fn']:
#                fit_idx.append(i)
#                log.info('Found module %d (%s) for subset prefit', i, fn)
#        tmodelspec.append(m)

    if type(fit_set[0]) is int:
        fit_idx = fit_set
    else:
        fit_idx = []
        for i, m in enumerate(modelspec):
            for fn in fit_set:
                if fn in m['fn']:
                    fit_idx.append(i)
                    log.info('Found module %d (%s) for subset prefit', i, fn)

    tmodelspec = copy.deepcopy(modelspec)



    if len(fit_idx) == 0:
        log.info('No modules matching fit_set for subset prefit')
        return modelspec

    exclude_idx = np.setdiff1d(np.arange(0, len(modelspec)),
                               np.array(fit_idx)).tolist()
    for i in exclude_idx:
        m = tmodelspec[i]
        if not m.get('phi'):
            log.info('Intializing phi for module %d (%s)', i, m['fn'])
            m = priors.set_mean_phi([m])[0]  # Inits phi

        log.info('Freezing phi for module %d (%s)', i, m['fn'])

        m['fn_kwargs'].update(m['phi'])
        m['phi'] = {}
        tmodelspec[i] = m

    # fit the subset of modules
    if metric is None:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       fit_kwargs=fit_kwargs)
    else:
        tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                       metric=metric, fit_kwargs=fit_kwargs)

    # reassemble the full modelspec with updated phi values from tmodelspec

    for i in fit_idx:
        modelspec[i] = tmodelspec[i]

    return modelspec


def init_dexp(rec, modelspec, nl_mode=2):
    """
    choose initial values for dexp applied after preceeding fir is
    initialized
    """
    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    target_i = find_module('double_exponential', modelspec)
    if target_i is None:
        log.warning("No dexp module was found, can't initialize.")
        return modelspec

    if target_i == len(modelspec):
        fit_portion = modelspec
    else:
        fit_portion = modelspec[:target_i]

    # ensures all previous modules have their phi initialized
    # choose prior mean if not found
    for i, m in enumerate(fit_portion):
        if ('phi' not in m.keys()) and ('prior' in m.keys()):
            log.debug('Phi not found for module, using mean of prior: %s',
                      m)
            m = priors.set_mean_phi([m])[0]  # Inits phi for 1 module
            fit_portion[i] = m

    # generate prediction from module preceeding dexp
    ms.fit_mode_on(fit_portion)
    rec = ms.evaluate(rec, fit_portion)
    ms.fit_mode_off(fit_portion)

    in_signal = modelspec[target_i]['fn_kwargs']['i']
    pchans = rec[in_signal].shape[0]
    amp = np.zeros([pchans, 1])
    base = np.zeros([pchans, 1])
    kappa = np.zeros([pchans, 1])
    shift = np.zeros([pchans, 1])

    for i in range(pchans):
        resp = rec['resp'].as_continuous()
        pred = rec[in_signal].as_continuous()[i:(i+1), :]
        if resp.shape[0] == pchans:
            resp = resp[i:(i+1), :]

        keepidx = np.isfinite(resp) * np.isfinite(pred)
        resp = resp[keepidx]
        pred = pred[keepidx]

        # choose phi s.t. dexp starts as almost a straight line
        # phi=[max_out min_out slope mean_in]
        # meanr = np.nanmean(resp)
        stdr = np.nanstd(resp)

        # base = np.max(np.array([meanr - stdr * 4, 0]))
        base[i, 0] = np.min(resp)
        # base = meanr - stdr * 3

        # amp = np.max(resp) - np.min(resp)
        if nl_mode == 1:
            amp[i, 0] = stdr * 3
            predrange = 2 / (np.max(pred) - np.min(pred) + 1)
        elif nl_mode == 2:
            mask=np.zeros_like(pred,dtype=bool)
            pct=91
            while sum(mask)<.01*pred.shape[0]:
                pct-=1
                mask=pred>np.percentile(pred,pct)
            if pct !=90:
                log.warning('Init dexp: Default for init mode 2 is to find mean '
                         'of responses for times where pred>pctile(pred,90). '
                         '\nNo times were found so this was lowered to '
                         'pred>pctile(pred,%d).', pct)
            amp[i, 0] = resp[mask].mean()
            predrange = 2 / (np.std(pred)*3)
        else:
            raise ValueError('nl mode = {} not valid'.format(nl_mode))
        
        shift[i, 0] = np.mean(pred)
        # shift = (np.max(pred) + np.min(pred)) / 2

        kappa[i, 0] = np.log(predrange)
        
    modelspec[target_i]['phi'] = {'amplitude': amp, 'base': base,
                                  'kappa': kappa, 'shift': shift}
    log.info("Init dexp: %s", modelspec[target_i]['phi'])

    return modelspec


def init_logsig(rec, modelspec):
    '''
    Initialization of priors for logistic_sigmoid,
    based on process described in methods of Rabinowitz et al. 2014.
    '''
    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    target_i = find_module('logistic_sigmoid', modelspec)
    if target_i is None:
        log.warning("No logsig module was found, can't initialize.")
        return modelspec

    if target_i == len(modelspec):
        fit_portion = modelspec
    else:
        fit_portion = modelspec[:target_i]

    # generate prediction from module preceeding dexp
    ms.fit_mode_on(fit_portion)
    rec = ms.evaluate(rec, fit_portion)
    ms.fit_mode_off(fit_portion)

    pred = rec['pred'].as_continuous()
    resp = rec['resp'].as_continuous()

    mean_pred = np.nanmean(pred)
    min_pred = np.nanmean(pred) - np.nanstd(pred)*3
    max_pred = np.nanmean(pred) + np.nanstd(pred)*3
    if min_pred < 0:
        min_pred = 0
        mean_pred = (min_pred+max_pred)/2

    pred_range = max_pred - min_pred
    min_resp = max(np.nanmean(resp)-np.nanstd(resp)*3, 0)  # must be >= 0
    max_resp = np.nanmean(resp)+np.nanstd(resp)*3
    resp_range = max_resp - min_resp

    # Rather than setting a hard value for initial phi,
    # set the prior distributions and let the fitter/analysis
    # decide how to use it.
    base0 = min_resp + 0.05*(resp_range)
    amplitude0 = resp_range
    shift0 = mean_pred
    kappa0 = pred_range
    log.info("Initial   base,amplitude,shift,kappa=({}, {}, {}, {})"
             .format(base0, amplitude0, shift0, kappa0))

    base = ('Exponential', {'beta': base0})
    amplitude = ('Exponential', {'beta': amplitude0})
    shift = ('Normal', {'mean': shift0, 'sd': pred_range})
    kappa = ('Exponential', {'beta': kappa0})

    modelspec[target_i]['prior'].update({
            'base': base, 'amplitude': amplitude, 'shift': shift,
            'kappa': kappa})

    modelspec[target_i]['bounds'] = {
            'base': (1e-15, None),
            'amplitude': (1e-15, None),
            'shift': (None, None),
            'kappa': (1e-15, None)
            }

    return modelspec


def fir_L2_norm(modelspec):
    modelspec = copy.deepcopy(modelspec)
    fir_idx = find_module('fir', modelspec)
    prior = priors._tuples_to_distributions(modelspec[fir_idx]['prior'])
    random_coeffs = np.random.rand(*prior['coefficients'].mean().shape)
    normed = random_coeffs / np.linalg.norm(random_coeffs)
    # Assumes fir phi hasn't been initialized yet and that coefficients
    # is the only parameter to set. MAY NOT BE TRUE FOR SOME MODELS.
    modelspec[fir_idx]['phi'] = {'coefficients': normed}

    return modelspec
