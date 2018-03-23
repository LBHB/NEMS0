import logging
log = logging.getLogger(__name__)

import copy

from nems.utils import split_keywords
from nems import keywords
from nems.fitters.api import scipy_minimize

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
        if 'i' in modelspec[i]['fn_kwargs'].keys() and modelspec[i]['fn_kwargs']['i']=='resp':
            # psth-based prediction, never use stim, just feed resp to pred
            first_input_to_stim=True
        elif 'i' in modelspec[i]['fn_kwargs'].keys() and modelspec[i]['fn_kwargs']['i']=='pred':
            modelspec[i]['fn_kwargs']['i']='stim'
            first_input_to_stim=True
        i+=1

    # insert metadata, if provided
    if not 'meta' in modelspec[0].keys():
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
                     fitter=scipy_minimize, fit_kwargs={}):
    """Removes all modules from the modelspec that come after the
    first occurrence of the target module, then performs a
    rough fit on the shortened modelspec, then adds the latter
    modules back on and returns the full modelspec.
    """
    target_i = None
    for i, m in enumerate(modelspec):
        if target_module in m['fn']:
            target_i = i+1
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
        nonfit_portion = []
    else:
        fit_portion = modelspec[:target_i]
        nonfit_portion = modelspec[target_i:]

    modelspec = analysis_function(rec, fit_portion, fitter=fitter,
                                  fit_kwargs=fit_kwargs)[0]
    modelspec.extend(nonfit_portion)
    return modelspec
