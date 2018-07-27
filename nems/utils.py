import time
import numpy as np

import logging
log = logging.getLogger(__name__)


def iso8601_datestring():
    '''
    Returns a string containing the present date as a string.
    '''
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


def split_to_api_and_fn(mystring):
    '''
    Returns (api, fn_name) given a string that would be used to import
    a function from a package.
    '''
    matches = mystring.split(sep='.')
    api = '.'.join(matches[:-1])
    fn_name = matches[-1]
    return api, fn_name


def shrinkage(mH, eH, sigrat=1, thresh=0):
    """
    apply shrinkage transformation to estimated mean value (mH),
    based on the relative size of the standard error (eH)
    """

    smd = np.abs(mH) / (eH + np.finfo(float).eps * (eH == 0)) / sigrat

    if thresh:
        hf = mH * (smd > 1)
    else:
        smd = 1 - np.power(smd, -2)
        smd = smd * (smd > 0)
        # smd[np.isnan(smd)]=0
        hf = mH * smd

    return hf


def progress_fun():
    """
    This function can be redirected to a function that tracks progress
    externallly, eg, in a queueing system
    """
    pass


def find_module(query, modelspec, find_all_matches=False, key='fn'):
    """
    name : string
    modelspec : NEMS modelspec (list of dictionaries)

    find_all_matches : boolean
      if True:
          returns first index where modelspec[]['fn'] contains name
          returns None if no match
      if False
          returns list of index(es) where modelspec[]['fn'] contains name
          returns empty list [] if no match
    """
    if find_all_matches:
        target_i = []
    else:
        target_i = None
    for i, m in enumerate(modelspec):
        if query in m[key]:
            if find_all_matches:
                target_i.append(i)
            else:
                target_i = i
                break

    if not target_i:
        log.debug("target_module: %s not found in modelspec.", query)
    else:
        log.debug("target_module: %s found at modelspec[%d]",
                  query, target_i)

    return target_i


def escaped_split(string, delimiter):
    '''
    Allows escaping of characters when splitting a string to a list,
    useful for some arguments in keyword strings that need to use
    underscores, decimals, hyphens, or other characters parsed by
    the keyword system.
    '''
    x = 'EXTREMELYUNLIKELYTOEVERENCOUNTERTHISEXACTSTRINGANYWHEREELSE'
    match = "\%s" % delimiter
    temp = string.replace(match, x)
    temp_split = temp.split(delimiter)
    final_split = [s.replace(x, match) for s in temp_split]

    return final_split


def escaped_join(list, delimiter):
    '''
    Allows escaping of characters when joining a list of strings,
    useful for some arguments in keyword strings that need to use
    underscores, decimals, hyphens, or other characters parsed by
    the keyword system.
    '''
    x = 'EXTREMELYUNLIKELYTOEVERENCOUNTERTHISEXACTSTRINGANYWHEREELSE'
    match = "\%s" % delimiter
    temp = [s.replace(match, x) for s in list]
    temp_join = delimiter.join(temp)
    final_join = temp_join.replace(x, match)

    return final_join

