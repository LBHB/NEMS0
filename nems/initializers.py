import logging
log = logging.getLogger(__name__)

import copy
import numpy as np

from nems.utils import split_keywords
from nems import keywords
from nems.fitters.api import scipy_minimize
import nems.modelspec as ms

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
    # can't hard code in keywords, since we don't know which keyword will be first.
    # and can't assume that it will be module[0] because those might be
    # state manipulations
    first_input_to_stim=False
    i=0
    while not first_input_to_stim and i<len(modelspec):
#        if 'i' in modelspec[i]['fn_kwargs'].keys() and modelspec[i]['fn_kwargs']['i']=='resp':
#            # psth-based prediction, never use stim, just feed resp to pred
#            first_input_to_stim=True
        if 'i' in modelspec[i]['fn_kwargs'].keys() and modelspec[i]['fn_kwargs']['i']=='pred':
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
            target_i = i+1
            break

    if not target_i:
        log.info("target_module: {} not found in modelspec.".format(target_module))
        return modelspec
    else:
        log.info("target_module: {0} found at modelspec[{1}]."
                             .format(target_module, target_i-1))

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    exclude_idx = []
    for i, m in enumerate(modelspec):
        for fn in extra_exclude:
            if fn in m['fn']:
                exclude_idx.append(i)
                log.info("excluding {0} from prefit".format(fn))

    # find modeules to keep (before target_i and not exlcuded)
    fitidx = np.setdiff1d(np.arange(target_i), np.array(exclude_idx))
    tmodelspec = []
    for i in fitidx:
        tmodelspec.append(modelspec[i])
    if fitidx[0] > 0 and modelspec[0]['fn_kwargs']['i'] == 'stim':
        tmodelspec[0]['fn_kwargs']['i'] = 'stim'

    # fit the subset of modules
    tmodelspec = analysis_function(rec, tmodelspec, fitter=fitter,
                                   fit_kwargs=fit_kwargs)[0]

    # reassemble the full modelspec with updated phi values from tmodelspec
    if fitidx[0] > 1 and modelspec[0]['fn_kwargs']['i'] == 'stim':
        tmodelspec[0]['fn_kwargs']['i'] == 'pred'
    for i, j in enumerate(fitidx):
        modelspec[j] = tmodelspec[i]

    return modelspec

def init_dexp(rec, modelspec):
    """
    choose initial values for dexp applied after preceeding fir is
    initialized
    """
    target_i = None
    target_module='double_exponential'
    for i, m in enumerate(modelspec):
        if target_module in m['fn']:
            target_i = i
            break

    if not target_i:
        log.info("target_module: {} not found in modelspec."
                             .format(target_module))
        return modelspec
    else:
        log.info("target_module: {0} found at modelspec[{1}]."
                             .format(target_module,target_i-1))

    if target_i == len(modelspec):
        fit_portion = modelspec
    else:
        fit_portion = modelspec[:target_i]

    # generate prediction from module preceeding dexp
    rec=ms.evaluate(rec,fit_portion)
    resp = rec['resp'].as_continuous()
    pred = rec['pred'].as_continuous()
    keepidx = np.isfinite(resp) * np.isfinite(pred)
    resp = resp[keepidx]
    pred = pred[keepidx]

    # choose phi s.t. dexp starts as almost a straight line
    # phi=[max_out min_out slope mean_in]
    meanr = np.nanmean(resp)
    stdr = np.nanstd(resp)
    modelspec[target_i]['phi']={}
    modelspec[target_i]['phi']['amplitude']=stdr * 8
    modelspec[target_i]['phi']['base']=meanr - stdr * 4
    modelspec[target_i]['phi']['kappa']=np.log(np.std(pred) / 10)
    modelspec[target_i]['phi']['shift']=np.mean(pred)
    log.info(modelspec[target_i])

    return modelspec

